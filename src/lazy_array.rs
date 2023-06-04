use std::marker::PhantomData;

use hdf5::Dataset;
use hdf5::H5Type;

use ndarray::Array;
use ndarray::ArrayView;
use ndarray::Dimension;
use ndarray::SliceArg;

use crate::error;

/// one dimensional lazy array backed by HDF5 dataset
pub type LazyArray1<T> = LazyArray<T, ndarray::Ix1>;
/// two dimensional lazy array backed by HDF5 dataset
pub type LazyArray2<T> = LazyArray<T, ndarray::Ix2>;
/// three dimensional lazy array backed by HDF5 dataset
pub type LazyArray3<T> = LazyArray<T, ndarray::Ix3>;
/// four dimensional lazy array backed by HDF5 dataset
pub type LazyArray4<T> = LazyArray<T, ndarray::Ix4>;
/// five dimensional lazy array backed by HDF5 dataset
pub type LazyArray5<T> = LazyArray<T, ndarray::Ix5>;
/// six dimensional lazy array backed by HDF5 dataset
pub type LazyArray6<T> = LazyArray<T, ndarray::Ix6>;

/// An object that can be treated similar to an [`ndarray::Array`],
/// but backed by an on-disk HDF5 dataset instead of an in-memory array.
///
/// `T` is the datatype of the underlying array (`f32`, `f64`, `usize`, etc),
/// and `DIM` is the dimension of the data. Consider using a type alias such  
/// as [`LazyArray3`], [`LazyArray4`], etc to skip specifying the dimension.
///
/// `LazyArray` objects are type-checked and dimension-checked when constructed
/// with [`LazyArray::new`]
pub struct LazyArray<T, DIM>
where
    DIM: Dimension,
    T: H5Type,
{
    dataset: Dataset,
    name: String,
    _numeric_type: PhantomData<T>,
    _dimension: PhantomData<DIM>,
}

impl<T, DIM> LazyArray<T, DIM>
where
    DIM: Dimension,
    T: H5Type,
{
    /// generate a dimension-checked `LazyArray` on t
    fn new(dataset: Dataset) -> Result<Self, crate::Error> {
        // first, try to do a full size read to ensure the dimension is correct
        let ndim: usize = DIM::NDIM.expect("no dynamically sized arrays at this time");

        let name = dataset.name();

        // check the dimensions are correct
        if ndim != dataset.ndim() {
            return Err(error::DimensionMismatch::new(&name, ndim, dataset.ndim()).into());
        }

        // check the datatypes are correct
        let dtype = dataset
            .dtype()
            .map_err(|e| error::MissingDatatype::from_field_name(&name, e))?;

        if !dtype.is::<T>() {
            return Err(error::WrongDatatype::new(&name, dtype).into());
        }

        let ret = Self {
            dataset,
            name,
            _numeric_type: PhantomData,
            _dimension: PhantomData,
        };

        Ok(ret)
    }

    /// read a slice of data from the HDF5 dataset.
    ///
    /// `info` is normally constructed with the [`ndarray::s`] macro, and slices into the dataset
    /// as it would slice into a [`ndarray::ArrayBase`] object. Usually this method 
    /// does not fail as the dimension and type have been checked previously
    pub(crate) fn slice<I>(&self, info: I) -> Result<Array<T, I::OutDim>, crate::Error>
    where
        I: SliceArg<DIM>,
        hdf5::Selection: From<I>,
    {
        self.dataset
            .read_slice(info)
            .map_err(|e| error::ReadSlice::from_field_name(&self.name, e).into())
    }

    /// write an array slice to a the underlying HDF5 dataset
    ///
    /// `info` is normally constructed with the [`ndarray::s`] macro, and slices into the dataset
    /// as it would slice into a [`ndarray::ArrayBase`] object. Usually this method 
    /// does not fail as the dimension and type have been checked previously
    pub(crate) fn write_slice<'a, ARR, I>(&self, array: ARR, info: I) -> Result<(), crate::Error>
    where
        ArrayView<'a, T, DIM>: From<ARR>,
        I: SliceArg<DIM>,
        hdf5::Selection: From<I>,
    {
        self.dataset
            .write_slice(array, info)
            .map_err(|e| crate::Error::from(error::WriteSlice::from_field_name(&self.name, e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn construct_array() {
        todo!()
    }

    #[test]
    fn slice_array() {
        todo!()
    }

    #[test]
    fn write_slice() {
        todo!()
    }
}
