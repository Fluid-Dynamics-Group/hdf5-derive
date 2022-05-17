#![doc = include_str!("../README.md")]

/// Derive read and write capabilities for a struct of arrays
///
/// refer to the [crate-level](index.html) documentation
pub use macros::HDF5;

pub use hdf5::File;
mod error;

pub use error::*;

#[derive(thiserror::Error, Debug)]
/// General error type that provides helpful information on what went wrong
pub enum Error {
    #[error(transparent)]
    /// Error when attempting to read a dataset that does not exist in an hdf5 file
    MissingDataset(#[from] error::MissingDataset),
    #[error(transparent)]
    /// Error when attempting to read an array to a given struct type
    ///
    /// This error can occur if the dimension of the h5 array does not match the dimension of the type
    /// provided
    SerializeArray(#[from] error::SerializeArray),
    #[error(transparent)]
    /// Failed to serialize an array to a hdf5 dataset
    WriteArray(#[from] error::WriteArray),
    #[error(transparent)]
    /// Could not create a dataset in a hdf5 file when writing
    CreateDataset(#[from] error::CreateDataset),
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
        #[hdf5(transpose="write")]
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
