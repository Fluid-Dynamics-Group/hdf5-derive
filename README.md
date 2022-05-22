# hdf5-derive

read and write arrays from an hdf5 file to a struct

## Usage

You can use the `hdf5_derive::HDF5` derive macro to make you struct of `ndarray::Array` data writeable. The methods
`.read_hdf5(file)` and `.write_hdf5(&self, file)` are automatically generated for you:

```
use hdf5_derive::{HDF5, ContainerIo};
use ndarray::Array3;
use ndarray::Array4;

let N = 100;

#[derive(HDF5)]
struct Data {
	pressure: Array3<f64>,
	velocity: Array4<f64>,
	temperature: Array3<f64>,
}

// fill the struct with some data
let data = Data {
	pressure: Array3::zeros((N, N, N)),
	temperature: Array3::zeros((N, N, N)),
	velocity: Array4::zeros((N, N, N, 3)),
};

// write data to a file
let file = hdf5_derive::File::create("some-file.h5").unwrap();
data.write_hdf5(&file);

// read data from a file
let loaded_data = Data::read_hdf5(&file);

std::fs::remove_file("some-file.h5").ok();
```

When passing a [`hdf5::File`](hdf5::File) to `read_hdf5` and `write_hdf5`, ensure that the 
file was opened with the correct permissions.

## Transposing

The rust [`ndarray`] library uses **row-major** (C-order) indexing to store its arrays. Row major indexing
implies that the **last index is the fastest** index to iterate over. Other common examples of row major (by default)
arrangements are:

* `ndarray`
* numpy
* C libraries

In contrast, **column-major** (Fortran-order) storage implies the **fastest index is the first index**. Common 
examples of column major indexing are:

* Fortran
* Matlab
* julia

Since `hdf5` has no sense of the order of the matrices stored, you must manually transpose the arrays read. You can do this through
the `transpose` attribute:

```
use hdf5_derive::HDF5;
use ndarray::Array3;

#[derive(HDF5)]
#[hdf5(transpose="read")]
struct FortranData {
	array: ndarray::Array3<u8>
}
```

The possible options for the transpose argument are:

* read
* write
* both
* none

You can override a container level attribute on a field as well:

```
use hdf5_derive::HDF5;
use ndarray::Array3;

#[derive(HDF5)]
#[hdf5(transpose="both")]
struct FortranData {
	#[hdf5(transpose="none")]
	// `array` here will not be transposed when read / written
	array: ndarray::Array3<usize>
}
```

## Renaming Arrays

By default, `hdf5_derive` looks for a dataset in the provided file with an identical name as the struct member.
You can use the `rename` attribute to change what should be read (or written) with a file:

```
use hdf5_derive::HDF5;
use ndarray::Array4;

#[derive(HDF5)]
struct RenamedData {
	#[hdf5(rename(write="my_array", read = "array_name_in_file"))]
	array: ndarray::Array3<usize>
}
```

or:

```
use hdf5_derive::HDF5;
use ndarray::Array4;

#[derive(HDF5)]
struct RenamedData {
	#[hdf5(rename(both = "my_array"))]
	array: ndarray::Array3<usize>
}
```

You can specify either `read`, `write`, `read` and `write`, or `both` if they `read == write`. If you 
specify `both` and `read` (or `write`), the value defaults to the expression provided in `both`.

## Mutating Existing Files

If you only wish to change some values from an existing file, then you can use the `#[mutate_on_write]` attribute
to mutate existing datasets. By default, `mutate_on_write` is set to `false` - meaning that datasets will be 
created for every field. If this field already exists, then this will result in an error. 

You can use container level attributes or field level attributes to specify the write behavior for a field. A container
level attribute will change the default behavior of all fields, but a field level attribute will supersede any container
level attributes (similar to `#[transpose]`). To mutate all fields in a struct:

```
use hdf5_derive::HDF5;
use ndarray::Array2;

#[derive(HDF5)]
#[hdf5(mutate_on_write)]
struct MutateData {
	// a dataset named `array` is now expected to already exist
	array: ndarray::Array3<usize>
}
```

If you are mutating some fields of an `hdf5` file while creating new fields for others, you can mix and match 
`mutate_on_write` for your desired behavior:

```
use hdf5_derive::HDF5;
use ndarray::Array2;

#[derive(HDF5)]
#[hdf5(mutate_on_write)]
struct MutateData {
	// a dataset named `array` is now expected to already exist
	array: ndarray::Array3<usize>,

	// a dataset `create_me_dataset` will be created 
	#[hdf5(mutate_on_write=false)]
	create_me_dataset: ndarray::Array3<usize>
}
```

## Reading and writing large data files

`hdf5-derive` makes no attempt to partially load data from an array. Instead, the entire dataset specified is loaded
into memory. If you wish to only access a slice from a large file, it may be more efficient to directly use the `hdf5`
library.
