//! Error types for handling HDF5 data with additional context information

macro_rules! create_error_type {
    ($error_name:ident, $format_str:expr, $doc_str:expr) => {
        #[doc=$doc_str]
        #[derive(thiserror::Error, Debug)]
        #[error("{}\n\nsource:\n{}", .msg, .source)]
        pub struct $error_name {
            msg: String,
            #[source]
            source: hdf5::Error,
        }

        impl $error_name {
            /// construct an instance of this type using the name of the dataarray and
            /// the corresponding HDF5 error
            pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
                let msg = format!($format_str, name);
                Self { msg, source }
            }
        }
    };
}

create_error_type! {
    MissingDataset,
    "failed to read dataset with name `{}`",
    "Error when attempting to read a dataset that does not exist in an hdf5 file"
}

create_error_type! {
    SerializeArray,
    "dataset `{}` exists, but it could not be read to the array type provided",
    r#"Error when attempting to read an array to a given struct type.


This error can occur if the dimension of the h5 array does not match the dimension of the type
provided"#
}

create_error_type! {
    CreateDataset,
    "Failed to create a dataset for `{}`",
    "Could not create a dataset in a hdf5 file when writing"
}

create_error_type! {
    CreateGroup,
    "Failed to create a group for `{}`",
    "Could not create a group in a hdf5 file when writing"
}

create_error_type! {
    MissingGroup,
    "Failed to read a group `{}`",
    "Could not open an existing group in a hdf5 file when writing"
}

create_error_type! {
    CreateAttribute,
    "Failed to create a attribute for `{}`",
    "Could not create a dataset in a hdf5 file when writing"
}

create_error_type! {
    WriteArray,
    "Failed to write array `{}` to dataset",
    "Failed to serialize an array to a hdf5 dataset"
}

create_error_type! {
    WriteAttribute,
    "Failed to write attribute `{}` to dataset",
    "Failed to serialize an attribute to a hdf5 dataset"
}

create_error_type! {
    FetchDataset,
    "Failed to fetch existing dataset `{}` when writing to file",
    "Could fetch an existing dataset when writing file"
}

create_error_type! {
    FetchAttribute,
    "Failed to fetch existing attribute `{}` when writing to file",
    "Could fetch an existing dataset when writing file"
}

create_error_type! {
    MissingAttribute,
    "Failed to fetch attribute `{}` when reading from file",
    "Could fetch an existing dataset when writing file"
}

create_error_type! {
    SerializeAttribute,
    "Failed to read attribute `{}` to the correct rust type",
    "Failed to serialize the dataset to the correct type after it has been written"
}

create_error_type! {
    MissingDatatype,
    "Datatype for array `{}` was missing / could not be fetched",
    "Failed to fetch the datatype of a given dataset"
}

create_error_type! {
    ReadSlice,
    "Failed to read slice for dataarray `{}`",
    "Failed to read a slice of data from an HDF5 dataset"
}

create_error_type! {
    WriteSlice,
    "Failed to write slice to dataarray `{}`",
    "Failed to write a slice of data to an HDF5 dataset"
}

#[derive(thiserror::Error, Debug)]
#[error("dimensions for array `{array_name}` were incorrect. Dataset was dimension `{dataset_dimension}`, not the specified dimension `{specified_dimension}`")]
/// [`crate::lazy_array::LazyArray`] underlying [`hdf5::Dataset`] was not the correct dimension
pub struct DimensionMismatch {
    array_name: String,
    dataset_dimension: usize,
    specified_dimension: usize,
}

impl DimensionMismatch {
    /// constructor for this type
    ///
    /// `array_name` is the dataset name that would back this lazy array
    pub fn new(array_name: &str, dataset_dimension: usize, specified_dimension: usize) -> Self {
        Self {
            array_name: array_name.into(),
            dataset_dimension,
            specified_dimension,
        }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("Specified datatype for array `{array_name}` was incorrect. Array type was {datatype:?}")]
/// Datatype for [`crate::LazyArray`] was incorrect
pub struct WrongDatatype {
    array_name: String,
    datatype: hdf5::datatype::Datatype,
}

impl WrongDatatype {
    /// constructor for this type
    ///
    /// `array_name` is the dataset name that would back this lazy array
    pub fn new(array_name: &str, datatype: hdf5::datatype::Datatype) -> Self {
        Self {
            array_name: array_name.into(),
            datatype,
        }
    }
}
