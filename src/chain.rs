use core::cmp::Ordering;
use core::fmt::{Debug, Display};
use std::collections::BTreeMap;
use std::error::Error;

use crate::{
    ctx::{AddCtx, ReadCtx},
    CmRDT, CvRDT, DotRange, VClock,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Op<V, A: Ord> {
    value: V,
    source: A,
    ctx: VClock<A>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Context<A: Ord>(VClock<A>);

impl<A: Ord + Clone> PartialOrd for Context<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Ord + Clone> Ord for Context<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ordering) => ordering,
            None => {
                let self_without_other = self.0.clone_without(&other.0);
                let other_without_self = other.0.clone_without(&self.0);

                let largest_self = self_without_other.dots.keys().rev().next();
                let largest_other = other_without_self.dots.keys().rev().next();
                match (largest_self, largest_other) {
                    (Some(self_actor), Some(other_actor)) => self_actor.cmp(&other_actor),
                    (Some(_), None) => Ordering::Greater,
                    (None, Some(_)) => Ordering::Less,
                    (None, None) => Ordering::Equal,
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Chain<V, A: Ord + Clone> {
    idempotency_clock: VClock<A>,
    context_clock: VClock<A>,
    chain: BTreeMap<Context<A>, V>,
}

#[derive(Debug, PartialEq, Eq)]
enum Validation<V, A: Ord> {
    ReusedContext {
        ctx: Context<A>,
        existing_value: V,
        op_value: V,
    },

    MissingDotRange(DotRange<A>),
}

impl<V, A: Ord> From<DotRange<A>> for Validation<V, A> {
    fn from(dot_range: DotRange<A>) -> Self {
        Self::MissingDotRange(dot_range)
    }
}

impl<V: Debug, A: Debug + Ord> Display for Validation<V, A> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&self, fmt)
    }
}

impl<V: Debug, A: Debug + Ord> Error for Validation<V, A> {}

impl<V: Debug + Clone + Eq, A: Debug + Ord + Clone> CmRDT for Chain<V, A> {
    type Op = Op<V, A>;
    type Validation = Validation<V, A>;

    fn validate_op(&self, op: &Self::Op) -> Result<(), Self::Validation> {
        let dot = op.ctx.inc(op.source.clone());
        self.idempotency_clock.validate_op(&dot)?;

        let mut ctx = Context(op.ctx.clone());
        ctx.0.apply(dot);

        if let Some(existing_value) = self.chain.get(&ctx) {
            if existing_value != &op.value {
                return Err(Validation::ReusedContext {
                    ctx,
                    existing_value: existing_value.clone(),
                    op_value: op.value.clone(),
                });
            }
        }
        Ok(())
    }

    /// Apply an Op to the CRDT
    fn apply(&mut self, op: Self::Op) {
        let Op { ctx, source, value } = op;
        if self.idempotency_clock.get(&source) >= ctx.get(&source) + 1 {
            // We've already seen this operation, dropping
        } else {
            let dot = ctx.inc(source);
            let mut chain_ctx = Context(ctx);
            chain_ctx.0.apply(dot.clone());

            self.idempotency_clock.apply(dot);
            self.context_clock.merge(chain_ctx.0.clone());
            self.chain.insert(chain_ctx, value);
        }
    }
}

impl<V: Eq, A: Ord + Clone + Debug> Chain<V, A> {
    pub fn append(&self, v: impl Into<V>, ctx: AddCtx<A>) -> Op<V, A> {
        Op {
            value: v.into(),
            source: ctx.actor,
            ctx: ctx.clock,
        }
    }

    pub fn value_ctx(&self, v: &V) -> Option<VClock<A>> {
        self.chain
            .iter()
            .find(|(_, value)| value == &v)
            .map(|(ctx, _)| ctx.0.clone())
    }

    pub fn read(&self) -> ReadCtx<Vec<&V>, A> {
        ReadCtx {
            add_clock: self.context_clock.clone(),
            rm_clock: self.context_clock.clone(),
            val: self.chain.values().collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::collections::VecDeque;

    use crate::quickcheck::{quickcheck, Arbitrary, Gen, TestResult};
    use crate::Dot;

    #[test]
    fn test_append() {
        let mut chain: Chain<u8, _> = Default::default();
        let read_ctx = chain.read();

        let append_op = chain.append(1, read_ctx.derive_add_ctx("A"));
        assert!(chain.validate_op(&append_op).is_ok());
        chain.apply(append_op);

        let read_ctx = chain.read();
        assert_eq!(read_ctx.val, vec![&1]);
        assert_eq!(read_ctx.add_clock, Dot::new("A", 1).into());

        let append_op = chain.append(2, read_ctx.derive_add_ctx("B"));
        assert_eq!(chain.validate_op(&append_op), Ok(()));
        chain.apply(append_op);

        let read_ctx = chain.read();
        assert_eq!(read_ctx.val, vec![&1, &2]);
        assert_eq!(
            read_ctx.add_clock,
            vec![Dot::new("A", 1), Dot::new("B", 1)]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn test_causal_appends() {
        let mut chain_a: Chain<u8, _> = Default::default();
        let mut chain_b: Chain<u8, _> = Default::default();

        let append_from_a = chain_a.append(0, chain_a.read().derive_add_ctx('A'));
        // TODO: add a CmRDT::validate_then_apply(op);
        assert_eq!(chain_b.validate_op(&append_from_a), Ok(()));
        chain_b.apply(append_from_a.clone());
        assert_eq!(chain_b.read().val, vec![&0]);

        let append_from_b_after_a = chain_b.append(1, chain_b.read().derive_add_ctx('B'));

        assert_eq!(chain_a.validate_op(&append_from_b_after_a), Ok(()));
        chain_a.apply(append_from_b_after_a.clone());
        assert_eq!(chain_a.read().val, vec![&1]);

        assert_eq!(chain_a.validate_op(&append_from_a), Ok(()));
        chain_a.apply(append_from_a);

        assert_eq!(chain_b.validate_op(&append_from_b_after_a), Ok(()));
        chain_b.apply(append_from_b_after_a);

        assert_eq!(chain_a, chain_b);
        assert_eq!(chain_a.read().val, vec![&0, &1]);
        assert_eq!(chain_b.read().val, vec![&0, &1]);
    }

    #[test]
    fn test_exchange_ops_before_applying_locally() {
        // let n = 2;
        // let instructions = vec![
        //     Instruction::Append { actor: 0, val: 1 },
        //     Instruction::Append { actor: 1, val: 0 },
        //     Instruction::Apply { dest: 0, source: 1 },
        // ];

        let mut chain_a: Chain<u8, _> = Default::default();
        let mut chain_b: Chain<u8, _> = Default::default();

        let append_from_a = chain_a.append(0, chain_a.read().derive_add_ctx('A'));
        let append_from_b = chain_b.append(1, chain_b.read().derive_add_ctx('B'));

        assert_eq!(chain_a.validate_op(&append_from_b), Ok(()));
        chain_a.apply(append_from_b.clone());
        assert_eq!(chain_b.validate_op(&append_from_a), Ok(()));
        chain_b.apply(append_from_a.clone());
        assert_eq!(chain_a.validate_op(&append_from_a), Ok(()));
        chain_a.apply(append_from_a);
        assert_eq!(chain_b.validate_op(&append_from_b), Ok(()));
        chain_b.apply(append_from_b);

        assert_eq!(chain_a, chain_b);
        assert_eq!(chain_a.read().val, vec![&0, &1]);
        assert_eq!(chain_b.read().val, vec![&0, &1]);
    }

    #[test]
    fn test_qc_stress_context_total_order() {
        let mut chain_a: Chain<u8, _> = Default::default();
        let mut chain_b: Chain<u8, _> = Default::default();
        let mut chain_c: Chain<u8, _> = Default::default();

        let append_from_b = chain_b.append(0, chain_b.read().derive_add_ctx('B'));
        let append_from_c = chain_c.append(1, chain_c.read().derive_add_ctx('C'));

        assert_eq!(chain_a.validate_op(&append_from_c), Ok(()));
        chain_a.apply(append_from_c.clone());
        assert_eq!(chain_a.read().val, vec![&1]);

        let append_from_a = chain_a.append(2, chain_a.read().derive_add_ctx('A'));

        assert_eq!(chain_a.validate_op(&append_from_a), Ok(()));
        chain_a.apply(append_from_a.clone());
        assert_eq!(chain_a.read().val, vec![&1, &2]);

        assert_eq!(chain_a.validate_op(&append_from_b), Ok(()));
        chain_a.apply(append_from_b.clone());
        assert_eq!(chain_a.read().val, vec![&0, &1, &2]);

        assert_eq!(chain_b.validate_op(&append_from_a), Ok(()));
        chain_b.apply(append_from_a.clone());
        assert_eq!(chain_b.read().val, vec![&2]);

        assert_eq!(chain_b.validate_op(&append_from_b), Ok(()));
        chain_b.apply(append_from_b.clone());
        assert_eq!(chain_b.read().val, vec![&0, &2]);

        assert_eq!(chain_b.validate_op(&append_from_c), Ok(()));
        chain_b.apply(append_from_c.clone());
        assert_eq!(chain_b.read().val, vec![&0, &1, &2]);

        assert_eq!(chain_c.validate_op(&append_from_a), Ok(()));
        chain_c.apply(append_from_a);
        assert_eq!(chain_c.read().val, vec![&2]);

        assert_eq!(chain_c.validate_op(&append_from_b), Ok(()));
        chain_c.apply(append_from_b);
        assert_eq!(chain_c.read().val, vec![&0, &2]);

        assert_eq!(chain_c.validate_op(&append_from_c), Ok(()));
        chain_c.apply(append_from_c);
        assert_eq!(chain_c.read().val, vec![&0, &1, &2]);

        assert_eq!(chain_a, chain_b);
        assert_eq!(chain_a, chain_c);
        assert_eq!(chain_b, chain_c);

        assert_eq!(chain_a.read().val, vec![&0, &1, &2]);
        assert_eq!(chain_b.read().val, vec![&0, &1, &2]);
        assert_eq!(chain_c.read().val, vec![&0, &1, &2]);
    }

    #[derive(Debug, Clone)]
    enum Instruction {
        Append { actor: u8, val: u8 },
        Apply { dest: u8, source: u8 },
    }

    impl Arbitrary for Instruction {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let source = u8::arbitrary(g) % 7;
            let dest = u8::arbitrary(g) % 7;
            let val = u8::arbitrary(g) % 5;

            match bool::arbitrary(g) {
                true => Self::Append {
                    actor: source,
                    val: val,
                },
                false => Self::Apply { dest, source },
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            match self.clone() {
                Self::Append { actor, val } => {
                    return Box::new(
                        (0..actor)
                            .into_iter()
                            .map(move |a| Self::Append { actor: a, val })
                            .chain(
                                (0..val)
                                    .into_iter()
                                    .map(move |v| Self::Append { actor, val: v }),
                            ),
                    );
                }

                Self::Apply { dest, source } => {
                    return Box::new(
                        (0..dest)
                            .into_iter()
                            .map(move |d| Self::Apply { dest: d, source })
                            .chain(
                                (0..source)
                                    .into_iter()
                                    .map(move |s| Self::Apply { dest, source: s }),
                            ),
                    );
                }
            }
        }
    }

    quickcheck! {
        fn prop_interpreter(n: usize, instructions: Vec<Instruction>) -> TestResult {
            if n == 0 || n >= 7 {
                return TestResult::discard();
            }

            let mut chains: Vec<_> = (0..n).map(|_| Chain::default()).collect();
            let mut op_queues: BTreeMap<usize, BTreeMap<usize, VecDeque<Op<u8, u8>>>> = Default::default();
            for instruction in instructions {
                match instruction {
                    Instruction::Append { actor, val } => {

                        let actor = (actor as usize) % n;

                        let append_in_progress = !op_queues.entry(actor)
                            .or_default()
                            .entry(actor)
                            .or_default()
                            .is_empty();

                        if append_in_progress {
                            continue
                        }

                        let chain = &chains[actor];
                        let add_ctx = chain.read().derive_add_ctx(actor as u8);
                        let op = chain.append(val, add_ctx);

                        for other_actor in 0..n {
                            op_queues.entry(other_actor)
                                .or_default()
                                .entry(actor)
                                .or_default()
                                .push_back(op.clone());
                        }
                    },
                    Instruction::Apply { dest, source } => {
                        let dest = (dest as usize) % n;
                        let source = (source as usize) % n;

                        let op_option = op_queues.entry(dest)
                            .or_default()
                            .entry(source)
                            .or_default()
                            .pop_front();

                        if let Some(op) = op_option {
                            assert_eq!(chains[dest].validate_op(&op), Ok(()));
                            chains[dest].apply(op);
                        }
                    }

                }
            }

            println!("Draining op queues {:#?}", op_queues);
            for (dest_actor, op_queue) in op_queues {
                for (_source, queue) in op_queue {
                    for op in queue {
                        assert_eq!(chains[dest_actor].validate_op(&op), Ok(()));
                        chains[dest_actor].apply(op);
                    }
                }
            }

            let mut chains_iter = chains.into_iter();
            let reference_chain = chains_iter.next().unwrap();

            for chain in chains_iter {
                assert_eq!(reference_chain, chain);
            }

            TestResult::passed()
        }

        fn prop_cmp_contexts(a: VClock<u8>, b: VClock<u8>) -> bool {
            let a_ctx = Context(a.clone());
            let b_ctx = Context(b.clone());

            let ordering = a_ctx.cmp(&b_ctx);
            if a == b {
                assert_eq!(ordering, Ordering::Equal);
            } else if a < b {
                assert_eq!(ordering, Ordering::Less);
            } else if a > b {
                assert_eq!(ordering, Ordering::Greater);
            } else {
                assert!(a.concurrent(&b));
                let a_without_b = a.clone_without(&b);
                let b_without_a = b.clone_without(&a);
                let a_dot = a_without_b.into_iter().last().unwrap();
                let b_dot = b_without_a.into_iter().last().unwrap();
                assert_eq!(a_dot.actor.cmp(&b_dot.actor), ordering)
            }
            true
        }

        fn prop_context_transitive(a: VClock<u8>, b: VClock<u8>, c: VClock<u8>) -> TestResult {
            let a = Context(a);
            let b = Context(b);
            let c = Context(c);

            if a < b && b < c {
                assert!(a < c);
            }
            if a == b && b == c {
                assert!(a == c);
            }
            if a > b && b > c {
                assert!(a > c);
            }

            TestResult::passed()
        }
    }
}

// TODO: replace `val` with `value`
