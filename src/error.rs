#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
/// Error when attempting to read a dataset that does not exist in an hdf5 file
pub struct MissingDataset {
    msg: String,
    #[source]
    source: hdf5::Error,
}

impl MissingDataset {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("failed to read dataset with name `{name}`");
        Self { msg, source }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
/// Error when attempting to read an array to a given struct type
///
/// This error can occur if the dimension of the h5 array does not match the dimension of the type
/// provided
pub struct SerializeArray {
    msg: String,
    #[source]
    source: hdf5::Error,
}

impl SerializeArray {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg =
            format!("dataset {name} exists, but it could not be read to the array type provided");
        Self { msg, source }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
/// Could not create a dataset in a hdf5 file when writing
pub struct CreateDataset {
    msg: String,
    #[source]
    source: hdf5::Error,
}

impl CreateDataset {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("Failed to create a dataset for {name}");
        Self { msg, source }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
/// Failed to serialize an array to a hdf5 dataset
pub struct WriteArray {
    msg: String,
    #[source]
    source: hdf5::Error,
}

impl WriteArray {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("Failed to write array `{name}` to dataset");
        Self { msg, source }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("{}", .msg)]
/// Could fetch an existing dataset when writing file
pub struct FetchDataset {
    msg: String,
    #[source]
    source: hdf5::Error,
}

impl FetchDataset {
    pub fn from_field_name(name: &str, source: hdf5::Error) -> Self {
        let msg = format!("Failed to fetch existing dataset `{name}` when writing to file");
        Self { msg, source }
    }
}
