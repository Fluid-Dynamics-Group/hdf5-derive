use std::marker::PhantomData;

use hdf5::Dataset;
use hdf5::H5Type;

use ndarray::Array;
use ndarray::ArrayView;
use ndarray::Dimension;
use ndarray::SliceArg;

use crate::error;
use crate::Error;

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
/// `LazyArray` objects are type-checked and dimension-checked when constructed
/// with [`LazyArray::new`]. `LazyArray`s are a nice abstraction over datasets
/// that ensure dataset errors happen at the time of creation, not at the time of processessing.
/// You should reach for a `LazyArray` when your dataset is otherwise too large to hold in memory.
/// Otherwise, the convenience of using `ndarray` backed arrays is generally nicer than HDF5 backed
/// arrays.
///
/// ## Macro Attributes
///
/// `LazyArray` ignores `transpose` and `mutate_on_write` attributes.
/// [`crate::ContainerWrite`]'s write routines have no effect on a `LazyArray`, all data is written
/// immediately to the underlying datasets.
///
/// `transpose` may have some effect in the future if required. For now, the following is ignored:
///
/// ```
/// use hdf5_derive::ContainerRead;
/// use hdf5_derive::LazyArray5;
///
/// #[derive(ContainerRead)]
/// struct Container {
///     #[hdf5(transpose="both")]
///     large_array: LazyArray5<u64>
/// }
/// ```
///
/// ## Types 
///
/// `T` is the datatype of the underlying array (`f32`, `f64`, `usize`, etc),
/// and `DIM` is the dimension of the data. Consider using a type alias such  
/// as [`LazyArray3`], [`LazyArray4`], etc to skip specifying the dimension.
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
    /// generate a dimension-checked and type-checked `LazyArray` for a given dataset. 
    /// Errors if the dimension of the dataset is different from the type, or if the type of the
    /// HDF5 data is different from the type.
    ///
    /// ## Exmaple
    ///
    /// ```
    /// let N = 500;
    /// let path = "./some_h5_file.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// // create a massive dataset, we would not want to hold this in memory
    /// let dataset = file
    ///     .new_dataset::<f64>()
    ///     .shape((N, N, N, N))
    ///     .create("some_dataset_name")
    ///     .unwrap();
    ///
    /// let lazy_array = hdf5_derive::LazyArray4::<f64>::new(dataset).unwrap();
    ///
    /// std::fs::remove_file(path);
    /// ```
    pub fn new(dataset: Dataset) -> Result<Self, crate::Error> {
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
    ///
    /// ## Exmaple
    ///
    /// ```
    /// use hdf5_derive::{LazyArray5, ContainerRead, ContainerWrite};
    /// let path = "./another_file.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// #[derive(ContainerWrite, ContainerRead)]
    /// struct WriteHelper {
    ///     arr: ndarray::Array5<f32>
    /// }
    ///
    /// let mut arr = ndarray::Array5::zeros((3,3,3,3,3));
    /// arr[[1,2,1,1,2]] = 10.;
    ///
    /// let helper= WriteHelper { arr: arr.clone() };
    /// helper.write_hdf5(&file);
    /// file.close();
    ///
    /// #[derive(ContainerWrite, ContainerRead)]
    /// struct LazyReader {
    ///     arr: LazyArray5<f32>
    /// }
    ///
    /// let file = hdf5_derive::File::open(path).unwrap();
    /// let lazy_reader = LazyReader::read_hdf5(&file).unwrap();
    ///
    /// // now we can slice the data directly from the h5 file without loading the 
    /// // entire file into memory!
    /// let slice = lazy_reader.arr.slice(ndarray::s![1,2,1,1,2]).unwrap();
    ///
    /// assert_eq!(slice.into_scalar(), 10.0);
    ///
    /// std::fs::remove_file(path);
    /// ```
    pub fn slice<I>(&self, info: I) -> Result<Array<T, I::OutDim>, crate::Error>
    where
        I: SliceArg<DIM> + TryInto<hdf5::Selection>,
        hdf5::Error: From<I::Error>, //hdf5::Selection: TryFrom<I>,
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
    ///
    /// ## Exmaple
    ///
    /// ```
    /// use hdf5_derive::{LazyArray5, ContainerRead, ContainerWrite};
    /// let path = "./another_file.h5";
    /// let file = hdf5_derive::File::create(path).unwrap();
    ///
    /// #[derive(ContainerWrite, ContainerRead)]
    /// struct WriteHelper {
    ///     arr: ndarray::Array5<f32>
    /// }
    ///
    /// let mut arr = ndarray::Array5::zeros((3,3,3,3,3));
    /// arr[[1,2,1,1,2]] = 10.;
    ///
    /// let helper= WriteHelper { arr: arr.clone() };
    /// helper.write_hdf5(&file);
    /// file.close();
    ///
    /// #[derive(ContainerWrite, ContainerRead)]
    /// struct LazyReader {
    ///     arr: LazyArray5<f32>
    /// }
    ///
    /// let file = hdf5_derive::File::open_rw(path).unwrap();
    /// let lazy_reader = LazyReader::read_hdf5(&file).unwrap();
    ///
    /// // the data was the same as how we wrote it
    /// let slice = lazy_reader.arr.slice(ndarray::s![1,2,1,1,2]).unwrap();
    /// assert_eq!(slice.into_scalar(), 10.0);
    ///
    /// // overwrite the data with a new slice
    /// let write_val = ndarray::arr0(3.0f32);
    /// let slice = lazy_reader.arr.write_slice(write_val.view(), ndarray::s![1,2,1,1,2]).unwrap();
    ///
    /// // go back and read the data, it should be different now
    /// let slice = lazy_reader.arr.slice(ndarray::s![1,2,1,1,2]).unwrap();
    /// assert_eq!(slice.into_scalar(), 3.0);
    ///
    /// std::fs::remove_file(path);
    /// ```
    pub fn write_slice<'a, ARR, I>(&self, array: ARR, info: I) -> Result<(), crate::Error>
    where
        ArrayView<'a, T, I::OutDim>: From<ARR>,
        I: SliceArg<DIM> + TryInto<hdf5::Selection>,
        hdf5::Error: From<I::Error>, //hdf5::Selection: TryFrom<I>,
    {
        self.dataset
            .write_slice(array, info)
            .map_err(|e| crate::Error::from(error::WriteSlice::from_field_name(&self.name, e)))?;

        Ok(())
    }

    /// provides access to the underlying dataset
    pub fn dataset(&self) -> &hdf5::Dataset {
        &self.dataset
    }
}

impl<T, DIM> std::ops::Deref for LazyArray<T, DIM>
where
    DIM: Dimension,
    T: H5Type,
{
    type Target = hdf5::Dataset;

    fn deref(&self) -> &Self::Target {
        &self.dataset
    }
}

impl<T, DIM> crate::ReadGroup for LazyArray<T, DIM>
where
    DIM: Dimension,
    T: H5Type,
{
    fn read_group(group: &hdf5::Group, array_name: &str, transpose: bool) -> Result<Self, Error>
    where
        Self: Sized,
    {
        if transpose {
            panic!("transposing not supported for LazyArray s");
        }

        let ds = group
            .dataset(array_name)
            .map_err(|e| error::MissingDataset::from_field_name(array_name, e))?;

        let ret = Self::new(ds)?;

        Ok(ret)
    }
}

impl<T, DIM> crate::WriteGroup for LazyArray<T, DIM>
where
    DIM: Dimension,
    T: H5Type,
{
    fn write_group(
        &self,
        _group: &hdf5::Group,
        _array_name: &str,
        transpose: bool,
        _mutate_on_write: bool,
    ) -> Result<(), Error>
    where
        Self: Sized,
    {
        if transpose {
            panic!("transpose not supported for writing lazy arrays");
        }

        // we do not need to do anything, all writes are already backed by the lazy array
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContainerRead, ContainerWrite};
    // required for proc macros to work
    use crate as hdf5_derive;

    use ndarray::s;
    use ndarray::Array3;

    #[derive(ContainerRead, ContainerWrite)]
    struct Helper {
        dim3: Array3<f32>,
    }

    #[derive(ContainerRead, ContainerWrite)]
    struct LazyTest {
        // this must be named the same as the other dataset!
        dim3: LazyArray3<f32>,
    }

    #[test]
    fn construct_array() {
        let path = "./lazy_construct_array.h5";
        let file = crate::File::create(path).unwrap();
        let data = ndarray::Array3::<f32>::zeros((5, 5, 5));

        let helper = Helper { dim3: data.clone() };
        helper.write_hdf5(&file).unwrap();
        file.close().unwrap();

        let file = crate::File::open(path).unwrap();

        // ensure we can construct
        let _lazy_reader = LazyTest::read_hdf5(&file).unwrap();

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn slice_array() {
        let path = "./lazy_slice_array.h5";
        let file = crate::File::create(path).unwrap();
        let mut data = ndarray::Array3::<f32>::zeros((5, 5, 5));

        data[[1, 1, 1]] = 10.;

        data[[3, 2, 1]] = 15.;

        // write some data to a file
        let helper = Helper { dim3: data.clone() };
        helper.write_hdf5(&file).unwrap();
        file.close().unwrap();

        // now, lazily read the data from the file
        let file = crate::File::open(path).unwrap();
        let lazy_reader = LazyTest::read_hdf5(&file).unwrap();

        let x = lazy_reader.dim3.slice(s![1, 1, 1]).unwrap().into_scalar();
        assert_eq!(x, 10.);

        let y = lazy_reader.dim3.slice(s![3, 2, 1]).unwrap().into_scalar();
        assert_eq!(y, 15.);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn write_slice() {
        let path = "./lazy_write_slice.h5";
        let file = crate::File::create(path).unwrap();
        let mut data = ndarray::Array3::<f32>::zeros((5, 5, 5));
        data[[0,0,0]] = 1.;

        let dataset = file
            .new_dataset::<f32>()
            .shape((5, 5, 5))
            .create("dim3")
            .unwrap();

        let lazy_writer = LazyArray::new(dataset).unwrap();
        let container = LazyTest { dim3: lazy_writer };
        container.dim3.write_slice(data.view(), s![.., .., ..]).unwrap();
        container.write_hdf5(&file).unwrap();
        file.close().unwrap();

        let file = crate::File::open(path).unwrap();
        let reader = Helper::read_hdf5(&file).unwrap();
        assert_eq!(data, reader.dim3);

        std::fs::remove_file(path).unwrap();
    }
}
