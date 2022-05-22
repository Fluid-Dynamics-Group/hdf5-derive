#![doc = include_str!("../README.md")]

/// Derive read and write capabilities for a struct of arrays
///
/// refer to the [crate-level](index.html) documentation
pub use macros::HDF5;

pub use hdf5::File;
pub use hdf5::Group;

mod error;

pub use error::*;

/// Provides methods for reading and writing to an [`hdf5`] file. Derived with [`HDF5`] macro.
pub trait ContainerIo {
    /// write the contents of a struct to an HDF5 file
    ///
    ///
    /// ```
    /// use hdf5_derive::ContainerIo;
    /// use hdf5_derive::HDF5;
    /// use ndarray::Array2;
    ///
    /// #[derive(HDF5)]
    /// struct Data {
    ///     some_field: Array2<u32>
    /// }
    ///
    /// let path = "./test_file_write.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// // write some data to the file
    /// let arr = Array2::zeros((5,5));
    /// let data = Data { some_field: arr.clone() };
    /// data.write_hdf5(&file).unwrap();
    ///
    /// // manually read the data using `hdf5` primitives
    /// let dset = file.dataset("some_field").unwrap();
    /// let read_array : Array2<u32> = dset.read().unwrap();
    ///
    /// // check that they are the same
    /// assert_eq!(read_array, arr);
    ///
    /// // remove this file for practical purposes
    /// std::fs::remove_file(path).unwrap();
    /// ```
    fn write_hdf5(&self, container: &File) -> Result<(), Error>;
    /// read the contents of an HDF5 file to `Self`
    ///
    /// ```
    /// use hdf5_derive::ContainerIo;
    /// use hdf5_derive::HDF5;
    /// use ndarray::Array2;
    ///
    /// #[derive(HDF5)]
    /// struct Data {
    ///     #[hdf5(rename(read = "some_field_renamed"))]
    ///     some_field: Array2<u32>
    /// }
    ///
    /// let path = "./test_file_write.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// // write some data to the file
    /// let arr = Array2::zeros((5,5));
    /// let dset = file.new_dataset::<u32>()
    ///     .shape((5,5))
    ///     .create("some_field_renamed")
    ///     .unwrap();
    /// dset.write(&arr).unwrap();
    ///
    /// // now, read the data from the written dataset
    /// let read_data = Data::read_hdf5(&file).unwrap();
    ///
    /// // check that they are the same
    /// assert_eq!(read_data.some_field, arr);
    ///
    /// // remove this file for practical purposes
    /// std::fs::remove_file(path).unwrap();
    /// ```
    fn read_hdf5(container: &Group) -> Result<Self, Error>
    where
        Self: Sized;
}

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
    #[error(transparent)]
    /// Could fetch an existing dataset when writing file
    FetchDataset(#[from] error::FetchDataset),
}

#[cfg(test)]
mod read_tests {
    use crate as hdf5_derive;
    use hdf5_derive::ContainerIo;
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
        let file = super::File::create(path).unwrap();

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
        let file = super::File::create(path).unwrap();

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
}

#[cfg(test)]
mod write_tests {
    use crate as hdf5_derive;
    use hdf5_derive::ContainerIo;
    use macros::HDF5;
    use std::fs;

    type Arr3 = ndarray::Array3<f64>;

    #[derive(HDF5)]
    struct TestWrite {
        one: Arr3,
    }

    #[test]
    fn simple_write_array() {
        let path = "simple_write_array.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
            .into_shape(shape)
            .unwrap();

        let x = TestWrite { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays are the same
        assert_eq!(x.one, new_arr);

        fs::remove_file(path).ok();
    }

    #[derive(HDF5)]
    struct TransposeWrite {
        #[hdf5(transpose = "write")]
        one: Arr3,
    }

    #[test]
    fn write_transposed() {
        let path = "write_transposed_1.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
            .into_shape(shape)
            .unwrap();

        let x = TransposeWrite { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays were transposed
        assert_eq!(x.one.t(), new_arr);

        fs::remove_file(path).ok();
    }

    #[derive(HDF5)]
    #[hdf5(transpose = "write")]
    struct TransposeWriteInherit {
        one: Arr3,
    }

    #[test]
    fn write_transposed_inherit() {
        let path = "write_transposed_inherit.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
            .into_shape(shape)
            .unwrap();

        let x = TransposeWriteInherit { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays were transposed
        assert_eq!(x.one.t(), new_arr);

        fs::remove_file(path).ok();
    }

    #[derive(HDF5)]
    #[hdf5(transpose = "write")]
    struct TransposeWriteOverride {
        #[hdf5(transpose = "none")]
        one: Arr3,
    }

    #[test]
    fn write_transposed_override() {
        let path = "write_transposed_override.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 20, 20);
        let arr = ndarray::Array::linspace(0., 100., shape.0 * shape.1 * shape.2)
            .into_shape(shape)
            .unwrap();

        let x = TransposeWriteOverride { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays are the same
        assert_eq!(x.one, new_arr);

        fs::remove_file(path).ok();
    }

    #[derive(HDF5)]
    struct ManuallyTransposedArray {
        #[hdf5(transpose = "write")]
        one: Arr3,
    }

    #[test]
    /// ensure non-standard layout arrays (non-contiguous) can be written without errors
    fn manually_transposed_array() {
        let path = "manually_transposed_array.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 4, 4);

        // write data out
        let arr = ndarray::Array::linspace(
            0.,
            (shape.0 * shape.1 * shape.2) as f64 - 1.,
            shape.0 * shape.1 * shape.2,
        )
        .into_shape(shape)
        .unwrap();

        // transpose the data outside of the library
        let arr = arr.t().to_owned();

        // the array is not standard layout
        assert!(arr.is_standard_layout() == false);

        let x = ManuallyTransposedArray { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays are the same
        assert_eq!(arr.t(), new_arr);

        fs::remove_file(path).ok();
    }

    #[derive(HDF5)]
    #[hdf5(mutate_on_write=true)]
    struct MutateOnWrite {
        one: Arr3,
    }

    #[test]
    fn mutate_on_write() {
        let path = "mutate_on_write.h5";
        fs::remove_file(path).ok();
        let file = super::File::create(path).unwrap();

        let shape = (5, 4, 4);

        // write data out
        let arr = ndarray::Array::linspace(
            0.,
            (shape.0 * shape.1 * shape.2) as f64 - 1.,
            shape.0 * shape.1 * shape.2,
        )
        .into_shape(shape)
        .unwrap();

         // create an existing dataset so we will error in .write_hdf5()
         // if we try to create it without mutating
         file
            .new_dataset::<f64>()
            .shape(shape)
            .create("one")
            .unwrap();

        let x = MutateOnWrite { one: arr.clone() };

        x.write_hdf5(&file).unwrap();

        let new_arr: Arr3 = file.dataset("one").unwrap().read().unwrap();

        // check the arrays are the same
        assert_eq!(arr, new_arr);

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
#[doc(hidden)]
pub trait ArrayType {
    type Ty;
}

impl<S, D> ArrayType for ndarray::ArrayBase<S, D>
where
    S: ndarray::RawData,
{
    type Ty = <S as ndarray::RawData>::Elem;
}
