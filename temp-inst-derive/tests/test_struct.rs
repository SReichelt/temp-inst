use std::pin::pin;

use temp_inst::{TempInst, TempInstPin, TempRef, TempRefMut};
use temp_inst_derive::{TempRepr, TempReprMut, TempReprMutChk};

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct EmptyStruct;

#[test]
fn empty_struct() {
    let _ = TempInst::<EmptyStruct>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct EmptyStruct2();

#[test]
fn empty_struct_2() {
    let _ = TempInst::<EmptyStruct2>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Unit(());

#[test]
fn unit() {
    let _ = TempInst::<Unit>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Unit2 {
    field0: (),
}

#[test]
fn unit_2() {
    let _ = TempInst::<Unit2>::new(());
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Single(TempRef<i32>);

#[test]
fn single() {
    let temp = TempInst::<Single>::new(&0);
    assert_eq!(*temp.0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Single2 {
    field0: TempRef<i32>,
}

#[test]
fn single_2() {
    let temp = TempInst::<Single2>::new(&0);
    assert_eq!(*temp.field0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct SingleMut(TempRefMut<i32>);

#[test]
fn single_mut() {
    let mut field0 = 0;
    let temp = pin!(TempInstPin::<SingleMut>::new(&mut field0));
    let temp = temp.deref_pin();
    assert_eq!(*temp.0, 0);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Multi(TempRef<i32>, TempRef<i32>);

#[test]
fn multi() {
    let temp = TempInst::<Multi>::new((&0, &1));
    assert_eq!(*temp.0, 0);
    assert_eq!(*temp.1, 1);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct Multi2 {
    field0: TempRef<i32>,
    field1: TempRef<i32>,
}

#[test]
fn multi_2() {
    let temp = TempInst::<Multi2>::new((&0, &1));
    assert_eq!(*temp.field0, 0);
    assert_eq!(*temp.field1, 1);
}

#[derive(TempRepr, TempReprMut, TempReprMutChk)]
struct MultiMut(TempRefMut<i32>, TempRefMut<i32>);

#[test]
fn multi_mut() {
    let mut field0 = 0;
    let mut field1 = 1;
    let temp = pin!(TempInstPin::<MultiMut>::new((&mut field0, &mut field1)));
    let temp = temp.deref_pin();
    assert_eq!(*temp.0, 0);
    assert_eq!(*temp.1, 1);
}
