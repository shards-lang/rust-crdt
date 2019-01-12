extern crate num_bigint;
use self::num_bigint::BigUint;
use crdts::GCounter;

#[test]
fn test_basic() {
    let mut a = GCounter::new();
    let mut b = GCounter::new();
    a.apply_inc("A".to_string());
    b.apply_inc("B".to_string());

    assert_eq!(a.read(), b.read());
    assert_ne!(a, b);

    a.apply_inc("A".to_string());

    assert_eq!(a.read(), b.read() + BigUint::from(1u8));
}