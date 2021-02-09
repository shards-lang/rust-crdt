use core::cmp::Ordering;
use core::fmt::{Debug, Display};
use std::collections::BTreeMap;
use std::error::Error;

use crate::{
    ctx::{AddCtx, ReadCtx},
    CmRDT, CvRDT, VClock,
};

struct Op<V, A: Ord> {
    value: V,
    ctx: VClock<A>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
struct Context<A: Ord>(VClock<A>);

impl<A: Ord + Clone> Ord for Context<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ordering) => ordering,
            None => {
                let self_without_other = self.0.clone_without(&other.0);
                let other_without_self = other.0.clone_without(&self.0);
                self_without_other.dots.cmp(&other_without_self.dots)
            }
        }
    }
}

#[derive(Debug, Default)]
struct Chain<V, A: Ord + Clone> {
    clock: VClock<A>,
    chain: BTreeMap<Context<A>, V>,
}

#[derive(Debug)]
enum Validation<V: Debug, A: Debug + Ord> {
    ReusedContext {
        ctx: VClock<A>,
        existing_value: V,
        op_value: V,
    },
}
impl<V: Debug, A: Debug + Ord> Display for Validation<V, A> {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&self, fmt)
    }
}

impl<V: Debug, A: Debug + Ord> Error for Validation<V, A> {}

impl<V: Debug + Clone + Eq, A: Debug + Ord + Clone> CmRDT for Chain<V, A> {
    type Op = Op<V, A>;
    /// The validation error returned by `validate_op`.
    type Validation = Validation<V, A>;

    fn validate_op(&self, op: &Self::Op) -> Result<(), Self::Validation> {
        match self.chain.get(&Context(op.ctx.clone())) {
            Some(existing_value) => {
                if existing_value != &op.value {
                    return Err(Validation::ReusedContext {
                        ctx: op.ctx.clone(),
                        existing_value: existing_value.clone(),
                        op_value: op.value.clone(),
                    });
                }
            }
            None => (),
        }

        Ok(())
    }

    /// Apply an Op to the CRDT
    fn apply(&mut self, op: Self::Op) {
        let Op { ctx, value } = op;
        if self.clock >= ctx {
            // We've already seen this operation, dropping
        } else {
            self.clock.merge(ctx.clone());
            self.chain.insert(Context(ctx), value);
        }
    }
}

impl<V: Eq, A: Ord + Clone> Chain<V, A> {
    fn append(&self, v: impl Into<V>, ctx: AddCtx<A>) -> Op<V, A> {
        Op {
            value: v.into(),
            ctx: ctx.clock,
        }
    }

    fn value_ctx(&self, v: &V) -> Option<VClock<A>> {
        self.chain
            .iter()
            .find(|(_, value)| value == &v)
            .map(|(ctx, _)| ctx.0.clone())
    }

    fn read(&self) -> ReadCtx<Vec<&V>, A> {
        ReadCtx {
            add_clock: self.clock.clone(),
            rm_clock: self.clock.clone(),
            val: self.chain.values().collect(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::quickcheck::{Arbitrary, Gen};
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
        assert!(chain.validate_op(&append_op).is_ok());
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

    // quickcheck! {
    //     fn prop_chain_respects_clock_order(contexts: Vec<VClock<u8>>) -> TestResult {
    //         let mut chain = Chain::default();

    //         for (val, ctx) in contexts.iter().cloned().enumerate() {
    //             let op = Op { val, ctx };
    //             if chain.validate_op(&op).is_err() {
    //                 return TestResult::discard();
    //             }
    //             chain.apply(&op)
    //         }

    //         let chain_vec = chain.read().val;

    //         for index in chain_vec {
    //             let index_ctx = chain.value_ctx(index).unwrap();
    //             for index in chain_vec {
    //                 let index_ctx = chain.value_ctx(index).unwrap();

    //             }
    //         }

    //     }
    // }
}

// TODO: replace `val` with `value`
