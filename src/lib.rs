#![doc = include_str!("../README.md")]

pub use macros::{ContainerRead, ContainerWrite};

pub use hdf5::File;
pub use hdf5::Group;

pub mod error;

#[doc(hidden)]
pub use error::*;

use num_traits::Zero;

/// Provides methods for writing a struct's contents to a file. Derived with [`ContainerWrite`]
/// proc macro.
pub trait ContainerWrite {
    /// write the contents of a struct to an HDF5 file
    ///
    ///
    /// ```
    /// use hdf5_derive::ContainerWrite;
    /// use ndarray::Array2;
    ///
    /// #[derive(ContainerWrite)]
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
    fn write_hdf5(&self, container: &Group) -> Result<(), Error>;
}

/// Provides methods for reading a struct's contents to a file. Derived with [`ContainerRead`]
/// proc macro.
pub trait ContainerRead {
    /// read the contents of an HDF5 file to `Self`
    ///
    /// ```
    /// use hdf5_derive::ContainerRead;
    /// use ndarray::Array2;
    ///
    /// #[derive(ContainerRead)]
    /// struct Data {
    ///     #[hdf5(rename(read = "some_field_renamed"))]
    ///     some_field: Array2<u32>
    /// }
    ///
    /// let path = "./test_file_read.h5";
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
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    MissingAttribute(#[from] error::MissingAttribute),
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    SerializeAttribute(#[from] error::SerializeAttribute),
    #[error(transparent)]
    /// Attribute was missing from hdf5 file
    FetchAttribute(#[from] error::FetchAttribute),
    #[error(transparent)]
    /// Could not create a attribute in a hdf5 file when writing
    CreateAttribute(#[from] error::CreateAttribute),
    #[error(transparent)]
    /// Could not create a attribute in a hdf5 file when writing
    WriteAttribute(#[from] error::WriteAttribute),
    #[error(transparent)]
    /// Could not open an existing group in a hdf5 file when writing
    MissingGroup(#[from] error::MissingGroup),
    #[error(transparent)]
    /// Could not create a group in a hdf5 file when writing
    CreateGroup(#[from] error::CreateGroup),
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

macro_rules! attributes{
    ($($scalar_type:ty),+) => {
        $(
            impl ReadGroup for $scalar_type {
                fn read_group(group: &Group, attribute_name: &str, _transpose: bool) -> Result<Self, Error> where Self: Sized {
                    let attribute_handle = group.attr(attribute_name)
                        .map_err(|e| error::MissingAttribute::from_field_name(attribute_name, e))?;

                    let attribute: Self = attribute_handle.read_scalar()
                        .map_err(|e| error::SerializeAttribute::from_field_name(attribute_name, e))?;

                    Ok(attribute)
                }
            }
            impl WriteGroup for $scalar_type {
                fn write_group(
                    &self,
                    group: &Group,
                    attribute_name: &str,
                    _transpose: bool,
                    mutate_on_write: bool,
                ) -> Result<(), Error>
                where
                    Self: Sized
                {
                    let hdf5_attribute = if mutate_on_write {
                        group.attr(attribute_name)
                            .map_err(|e| error::FetchAttribute::from_field_name(attribute_name, e))?
                    } else {
                        group.new_attr::<Self>()
                            .create(attribute_name)
                            .map_err(|e| error::CreateAttribute::from_field_name(attribute_name, e))?
                    };

                    hdf5_attribute.write_scalar(self)
                        .map_err(|e| error::WriteAttribute::from_field_name(attribute_name, e))?;


                    Ok(())
                }
            }
        )+
    }
}

attributes!(f32, f64, i16, i32, i64, i8, isize, u16, u8, u32, u64, usize);

/// Defines how a given piece of data should be parsed.
/// You likely do not want to use this trait; instead use the methods from [`ContainerRead`]
pub trait ReadGroup {
    fn read_group(group: &Group, array_name: &str, transpose: bool) -> Result<Self, Error>
    where
        Self: Sized;
}

impl<S, D> ReadGroup for ndarray::ArrayBase<ndarray::OwnedRepr<S>, D>
where
    S: hdf5::H5Type,
    D: ndarray::Dimension,
{
    fn read_group(group: &Group, array_name: &str, transpose: bool) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let dataset = group
            .dataset(array_name)
            .map_err(|e| MissingDataset::from_field_name(array_name, e))?;
        let output_array: Self = dataset
            .read()
            .map_err(|e| SerializeArray::from_field_name(array_name, e))?;

        // handle transposing the array
        let output_array = if transpose {
            output_array.reversed_axes()
        } else {
            output_array
        };

        Ok(output_array)
    }
}

impl<T> ReadGroup for T
where
    T: ContainerRead,
{
    fn read_group(group: &Group, container_name: &str, _transpose: bool) -> Result<Self, Error>
    where
        Self: Sized,
    {
        let subgroup: Group = group
            .group(container_name)
            .map_err(|e| error::MissingGroup::from_field_name(container_name, e))?;

        T::read_hdf5(&subgroup)
    }
}

/// Defines how a given piece of data should be written.
/// You likely do not want to use this trait; instead use the methods from [`ContainerWrite`]
pub trait WriteGroup {
    fn write_group(
        &self,
        group: &Group,
        array_name: &str,
        transpose: bool,
        mutate_on_write: bool,
    ) -> Result<(), Error>
    where
        Self: Sized;
}

impl<S, D> WriteGroup for ndarray::ArrayBase<ndarray::OwnedRepr<S>, D>
where
    S: hdf5::H5Type + Zero + Clone,
    D: ndarray::Dimension,
{
    fn write_group(
        &self,
        group: &Group,
        array_name: &str,
        transpose: bool,
        mutate_on_write: bool,
    ) -> Result<(), Error>
    where
        Self: Sized,
    {
        self.view()
            .write_group(group, array_name, transpose, mutate_on_write)
    }
}

impl<'a, S, D> WriteGroup for ndarray::ArrayBase<ndarray::ViewRepr<&'a S>, D>
where
    S: hdf5::H5Type + Zero + Clone,
    D: ndarray::Dimension,
{
    fn write_group(
        &self,
        group: &Group,
        array_name: &str,
        transpose: bool,
        mutate_on_write: bool,
    ) -> Result<(), Error>
    where
        Self: Sized,
    {
        // handle the case in which we have to transpose the array
        if transpose {
            let mut tmp = ndarray::Array::zeros(self.t().dim());
            tmp.assign(&self.t());

            return tmp.write_group(group, array_name, false, mutate_on_write);
        }

        let fetch_dataset = if mutate_on_write {
            // fetch an existing dataset that we can mutate
            group
                .dataset(array_name)
                .map_err(|e| error::FetchDataset::from_field_name(array_name, e))?
        } else {
            // create a new dataset
            group
                .new_dataset::<<Self as ArrayType>::Ty>()
                .shape(self.shape())
                .create(array_name)
                .map_err(|e| error::CreateDataset::from_field_name(array_name, e))?
        };

        fetch_dataset
            .write(self.view())
            .map_err(|e| error::WriteArray::from_field_name(array_name, e))?;

        Ok(())
    }
}

impl<T> WriteGroup for T
where
    T: ContainerWrite,
{
    fn write_group(
        &self,
        group: &Group,
        container_name: &str,
        _transpose: bool,
        mutate_on_write: bool,
    ) -> Result<(), Error>
    where
        Self: Sized,
    {
        let subgroup = if mutate_on_write {
            group
                .group(container_name)
                .map_err(|e| error::MissingGroup::from_field_name(container_name, e))?
        } else {
            group
                .create_group(container_name)
                .map_err(|e| error::CreateGroup::from_field_name(container_name, e))?
        };

        self.write_hdf5(&subgroup)?;

        Ok(())
    }
}
