use hdf5_derive::ContainerIo;
use hdf5_derive::File;
use macros::HDF5;
use std::fs;

type Arr3 = ndarray::Array3<f64>;

#[derive(HDF5)]
struct TestStruct {
    one: Arr3,
}

#[test]
fn simple_read_write() {
    let path = "simple_read_write.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 20, 20);

    // write data out
    let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
        .into_shape(shape)
        .unwrap();
    let dset = file
        .new_dataset::<f64>()
        .shape(shape)
        .create("one")
        .unwrap();
    dset.write(&arr).unwrap();

    // then parse the data
    let read_data = TestStruct::read_hdf5(&file).unwrap();

    // check the arrays are the same
    assert_eq!(read_data.one, arr);

    fs::remove_file(path).ok();
}

#[derive(HDF5)]
struct RenameArray {
    #[hdf5(rename(read = "two", write = "one"))]
    one: Arr3,
}

#[test]
fn simple_rename() {
    let path = "simple_rename.h5";
    fs::remove_file(path).ok();
    let file = File::create(path).unwrap();

    let shape = (5, 4, 4);

    // write data out
    let arr = ndarray::Array::linspace(
        0.,
        (shape.0 * shape.1 * shape.2) as f64 - 1.,
        shape.0 * shape.1 * shape.2,
    )
    .into_shape(shape)
    .unwrap();
    let dset = file
        .new_dataset::<f64>()
        .shape(shape)
        .create("two")
        .unwrap();
    dset.write(&arr).unwrap();

    // then parse the data
    let read_data = RenameArray::read_hdf5(&file).unwrap();

    // check the arrays are the same
    assert_eq!(read_data.one, arr);

    fs::remove_file(path).ok();
}
