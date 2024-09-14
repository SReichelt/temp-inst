use std::pin::pin;

use either::Either;
use temp_inst::{TempInst, TempInstPin, TempRef, TempRefMut};
use temp_inst_derive::{TempRepr, TempReprMut, TempReprMutChk};

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum EmptyEnum {}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumEmpty {
    Variant,
}

#[test]
fn single_enum_empty() {
    let _ = TempInst::<SingleEnumEmpty>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumEmpty2 {
    Variant(),
}

#[test]
fn single_enum_empty_2() {
    let _ = TempInst::<SingleEnumEmpty2>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumUnit {
    Variant(()),
}

#[test]
fn single_enum_unit() {
    let _ = TempInst::<SingleEnumUnit>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumUnit2 {
    Variant { field0: () },
}

#[test]
fn single_enum_unit_2() {
    let _ = TempInst::<SingleEnumUnit2>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumSingle {
    Variant0(TempRef<i32>),
}

#[test]
fn single_enum_single() {
    let temp = TempInst::<SingleEnumSingle>::new(&0);
    let SingleEnumSingle::Variant0(field0) = &*temp;
    assert_eq!(**field0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumSingle2 {
    Variant0 { field0: TempRef<i32> },
}

#[test]
fn single_enum_single_2() {
    let temp = TempInst::<SingleEnumSingle2>::new(&0);
    let SingleEnumSingle2::Variant0 { field0 } = &*temp;
    assert_eq!(**field0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumSingleMut {
    Variant0(TempRefMut<i32>),
}

#[test]
fn single_enum_single_mut() {
    let mut field0 = 0;
    let temp = pin!(TempInstPin::<SingleEnumSingleMut>::new(&mut field0));
    let SingleEnumSingleMut::Variant0(field0) = &*temp.deref_pin();
    assert_eq!(**field0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumMulti {
    Variant0(TempRef<i32>, TempRef<i32>),
}

#[test]
fn single_enum_multi() {
    let temp = TempInst::<SingleEnumMulti>::new((&0, &1));
    let SingleEnumMulti::Variant0(field0, field1) = &*temp;
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumMulti2 {
    Variant0 {
        field0: TempRef<i32>,
        field1: TempRef<i32>,
    },
}

#[test]
fn single_enum_multi_2() {
    let temp = TempInst::<SingleEnumMulti2>::new((&0, &1));
    let SingleEnumMulti2::Variant0 { field0, field1 } = &*temp;
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum SingleEnumMultiMut {
    Variant0(TempRefMut<i32>, TempRefMut<i32>),
}

#[test]
fn single_enum_multi_mut() {
    let mut field0 = 0;
    let mut field1 = 1;
    let temp = pin!(TempInstPin::<SingleEnumMultiMut>::new((
        &mut field0,
        &mut field1
    )));
    let SingleEnumMultiMut::Variant0(field0, field1) = &*temp.deref_pin();
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum DoubleEnum {
    Variant0(TempRef<i32>),
    Variant1(TempRef<i32>, TempRef<i32>, TempRef<i32>),
}

#[test]
fn double_enum_variant0() {
    let temp = TempInst::<DoubleEnum>::new(Either::Left(&0));
    let DoubleEnum::Variant0(field0) = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn double_enum_variant1() {
    let temp = TempInst::<DoubleEnum>::new(Either::Right((&0, &1, &2)));
    let DoubleEnum::Variant1(field0, field1, field2) = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum DoubleEnum2 {
    Variant0 {
        field0: TempRef<i32>,
    },
    Variant1 {
        field0: TempRef<i32>,
        field1: TempRef<i32>,
        field2: TempRef<i32>,
    },
}

#[test]
fn double_enum_2_variant0() {
    let temp = TempInst::<DoubleEnum2>::new(Either::Left(&0));
    let DoubleEnum2::Variant0 { field0 } = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn double_enum_2_variant1() {
    let temp = TempInst::<DoubleEnum2>::new(Either::Right((&0, &1, &2)));
    let DoubleEnum2::Variant1 {
        field0,
        field1,
        field2,
    } = &*temp
    else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum DoubleEnumMut {
    Variant0(TempRefMut<i32>),
    Variant1(TempRef<i32>, TempRef<i32>, TempRefMut<i32>),
}

#[test]
fn double_enum_mut_variant0() {
    let mut field0 = 0;
    let temp = pin!(TempInstPin::<DoubleEnumMut>::new(Either::Left(&mut field0)));
    let DoubleEnumMut::Variant0(field0) = &*temp.deref_pin() else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn double_enum_mut_variant1() {
    let mut field2 = 2;
    let temp = pin!(TempInstPin::<DoubleEnumMut>::new(Either::Right((
        &0,
        &1,
        &mut field2
    ))));
    let DoubleEnumMut::Variant1(field0, field1, field2) = &*temp.deref_pin() else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum TripleEnum {
    Variant0(TempRef<i32>),
    Variant1,
    Variant2(TempRef<i32>, TempRef<i32>, TempRef<i32>),
}

#[test]
fn triple_enum_variant0() {
    let temp = TempInst::<TripleEnum>::new(Either::Left(Either::Left(&0)));
    let TripleEnum::Variant0(field0) = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn triple_enum_variant1() {
    let temp = TempInst::<TripleEnum>::new(Either::Left(Either::Right(())));
    let TripleEnum::Variant1 = &*temp else {
        unreachable!()
    };
}

#[test]
fn triple_enum_variant2() {
    let temp = TempInst::<TripleEnum>::new(Either::Right((&0, &1, &2)));
    let TripleEnum::Variant2(field0, field1, field2) = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum TripleEnum2 {
    Variant0 {
        field0: TempRef<i32>,
    },
    Variant1,
    Variant2 {
        field0: TempRef<i32>,
        field1: TempRef<i32>,
        field2: TempRef<i32>,
    },
}

#[test]
fn triple_enum_2_variant0() {
    let temp = TempInst::<TripleEnum2>::new(Either::Left(Either::Left(&0)));
    let TripleEnum2::Variant0 { field0 } = &*temp else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn triple_enum_2_variant1() {
    let temp = TempInst::<TripleEnum2>::new(Either::Left(Either::Right(())));
    let TripleEnum2::Variant1 = &*temp else {
        unreachable!()
    };
}

#[test]
fn triple_enum_2_variant2() {
    let temp = TempInst::<TripleEnum2>::new(Either::Right((&0, &1, &2)));
    let TripleEnum2::Variant2 {
        field0,
        field1,
        field2,
    } = &*temp
    else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
enum TripleEnumMut {
    Variant0(TempRefMut<i32>),
    Variant1,
    Variant2(TempRef<i32>, TempRef<i32>, TempRefMut<i32>),
}

#[test]
fn triple_enum_mut_variant0() {
    let mut field0 = 0;
    let temp = pin!(TempInstPin::<TripleEnumMut>::new(Either::Left(
        Either::Left(&mut field0)
    )));
    let TripleEnumMut::Variant0(field0) = &*temp.deref_pin() else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
}

#[test]
fn triple_enum_mut_variant1() {
    let temp = pin!(TempInstPin::<TripleEnumMut>::new(Either::Left(
        Either::Right(())
    )));
    let TripleEnumMut::Variant1 = &*temp.deref_pin() else {
        unreachable!()
    };
}

#[test]
fn triple_enum_mut_variant2() {
    let mut field2 = 2;
    let temp = pin!(TempInstPin::<TripleEnumMut>::new(Either::Right((
        &0,
        &1,
        &mut field2
    ))));
    let TripleEnumMut::Variant2(field0, field1, field2) = &*temp.deref_pin() else {
        unreachable!()
    };
    assert_eq!(**field0, 0);
    assert_eq!(**field1, 1);
    assert_eq!(**field2, 2);
}
