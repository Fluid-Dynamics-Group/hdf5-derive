pub use hdf5::File;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    MissingDataset(#[from] MissingDataset),
    #[error(transparent)]
    SerializeArray(#[from] SerializeArray),
    #[error(transparent)]
    WriteArray(#[from] WriteArray),
    #[error(transparent)]
    CreateDataset(#[from] CreateDataset),
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
pub struct MissingDataset {
    msg: String,
    #[source]
    source: hdf5::Error
}

impl MissingDataset {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("failed to read dataset with name `{name}`");
        Self {msg, source}
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
pub struct SerializeArray {
    msg: String,
    #[source]
    source: hdf5::Error
}

impl SerializeArray {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("dataset {name} exists, but it could not be read to the array type provided");
        Self {msg, source}
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
pub struct CreateDataset {
    msg: String,
    #[source]
    source: hdf5::Error
}

impl CreateDataset {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("Failed to create a dataset for {name}");
        Self {msg, source}
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
pub struct WriteArray {
    msg: String,
    #[source]
    source: hdf5::Error
}

impl WriteArray {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("Failed to write array `{name}` to dataset");
        Self {msg, source}
    }
}

#[cfg(test)]
mod read_tests{
    use crate as hdf5_derive;
    use std::fs;
    use macros::HDF5;

    type Arr3 = ndarray::Array3<f64>;

    #[derive(HDF5)]
    struct TestStruct {
        one: Arr3
    }

    #[test]
    fn simple_read_write() {
        let path = "simple_read_write.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);

        // write data out
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2).into_shape(shape).unwrap();
        let dset = file.new_dataset::<f64>().shape(shape).create("one").unwrap();
        dset.write(&arr).unwrap();

        // then parse the data
        let read_data = TestStruct::read_hdf5(&file).unwrap();

        // check the arrays are the same
        assert_eq!(read_data.one, arr);

        fs::remove_file(path).ok();
    }
}

#[cfg(test)]
mod write_tests {
    use crate as hdf5_derive;
    use std::fs;
    use macros::HDF5;

    type Arr3 = ndarray::Array3<f64>;

    #[derive(HDF5)]
    struct TestWrite {
        one: Arr3
    }

    #[test]
    fn simple_write_array() {
        let path = "simple_write_array.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2).into_shape(shape).unwrap();

        let x = TestWrite { one: arr.clone()};

        x.write_hdf5(&file).unwrap();

        let new_arr : Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays are the same
        assert_eq!(x.one, new_arr);

        fs::remove_file(path).ok();
    }
}



/// Helper trait to determine the type of element that a given [`ArrayBase`](ndarray::ArrayBase)
/// implements
///
/// This trait is required because `hdf5` datasets require type information on the data they store
/// before creation, and the type system is not smart enough to determine this type based on the
/// subsequent writes.
///
/// Since the `ArrayBase` does not directly implement [`RawData`](ndarray::RawData), it is not
/// possible to determine the type of the array elements without a helper trait.
pub trait ArrayType {
    type Ty;
}

impl <S,D> ArrayType for ndarray::ArrayBase<S, D> 
    where S: ndarray::RawData
{
    type Ty = <S as ndarray::RawData>::Elem;
}

mod testing {
}
