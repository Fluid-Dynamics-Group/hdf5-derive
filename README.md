# hdf5-derive

read and write arrays from an hdf5 file to a struct

[Documentation](https://fluid-dynamics-group.github.io/hdf5-derive/hdf5_derive/index.html)

## Usage

You can use the `hdf5_derive::ContainerWrite` derive macro to make you struct of [`ndarray::Array<T>`](`ndarray::Array`) (or array view) data writeable. The macro
derives the [`ContainerWrite`] trait, which provides a `.write_hdf5` method. Conversely, you can make a struct readable from a hdf5 file with `hdf5_derive::ContainerRead`,
which derives the [`ContainerRead`] trait. Reference types cannot implement [`ContainerRead`].

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array3;
use ndarray::Array4;

let N = 100;

#[derive(ContainerRead, ContainerWrite)]
struct Data {
	pressure: Array3<f64>,
	velocity: Array4<f64>,
	temperature: Array3<f64>,
	reynolds_number: f64,
	timesteps: u64,
	supplemental_information: NestedData
}

#[derive(ContainerRead, ContainerWrite)]
struct NestedData {
	vorticity: Array4<f32>
}

// fill the struct with some data
let data = Data {
	pressure: Array3::zeros((N, N, N)),
	temperature: Array3::zeros((N, N, N)),
	velocity: Array4::zeros((N, N, N, 3)),
	timesteps: 10_000,
	reynolds_number: 1650.,
	supplemental_information: NestedData {
		vorticity: Array4::zeros((N,N,N,3))
	}
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

## Large Datasets

Sometimes, a dataset is too large to load into memory. By default, you have a container specified like this:

```
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array5;

#[derive(ContainerRead, ContainerWrite)]
struct LargeDataset {
	array_1: Array5<f64>,
	array_2: Array5<f64>,
	array_3: Array5<f64>,
}
```

all of the data for `array_1`, `array_2` and `array_3` will be loaded into memory. Since these arrays are 5 dimensional,
this may consume a large portion or memory. If you wish to avoid loading the array into memory, you can use
a [`LazyArray`] as a nice dimension-checked + type-checked abstraction over [`hdf5::Dataset`]. Lazy array
reads/writes are always to/from disk. The equivalent lazy code looks like this:

```
use hdf5_derive::{ContainerRead, ContainerWrite, LazyArray5};

#[derive(ContainerRead, ContainerWrite)]
struct LargeDataset {
	array_1: LazyArray5<f64>,
	array_2: LazyArray5<f64>,
	array_3: LazyArray5<f64>,
}
```

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

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array3;

#[derive(ContainerRead, ContainerWrite)]
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

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array3;

#[derive(ContainerRead, ContainerWrite)]
#[hdf5(transpose="both")]
struct FortranData {
	#[hdf5(transpose="none")]
	// `array` here will not be transposed when read / written
	array: ndarray::Array3<usize>
}
```

Note that a `#[transpose]` field attribute on a type implementing [`ContainerWrite`] or [`ContainerRead`]
(including derived types) will be ignored. Instead, the `#[transpose]` attributes on the struct or struct's fields will be
used instead.

## Renaming Arrays

By default, `hdf5_derive` looks for a dataset in the provided file with an identical name as the struct member.
You can use the `rename` attribute to change what should be read (or written) with a file:

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array4;

#[derive(ContainerRead, ContainerWrite)]
struct RenamedData {
	#[hdf5(rename(write="my_array", read = "array_name_in_file"))]
	array: ndarray::Array3<usize>
}
```

or:

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array4;

#[derive(ContainerRead, ContainerWrite)]
struct RenamedData {
	#[hdf5(rename(both = "my_array"))]
	array: ndarray::Array3<usize>
}
```

You can specify either `read`, `write`, `read` and `write`, or `both` if `read == write`. If you 
specify `both` and `read` (or `write`), the value defaults to the expression provided in `both`.

## Mutating Existing Files

If you only wish to change some values from an existing file, then you can use the `#[mutate_on_write]` attribute
to mutate existing datasets. By default, `mutate_on_write` is set to `false` - meaning that datasets will be 
created for every field. If this field already exists, then this will result in an error. 

You can use container level attributes or field level attributes to specify the write behavior for a field. A container
level attribute will change the default behavior of all fields, but a field level attribute will supersede any container
level attributes (similar to `#[transpose]`). To mutate all fields in a struct:

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};

#[derive(ContainerRead, ContainerWrite)]
#[hdf5(mutate_on_write)]
struct MutateData {
	// a dataset named `array` is now expected to already exist
	array: ndarray::Array3<usize>
}
```

If you are mutating some fields of an `hdf5` file while creating new fields for others, you can mix and match 
`mutate_on_write` for your desired behavior:

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array2;

#[derive(ContainerRead, ContainerWrite)]
#[hdf5(mutate_on_write)]
struct MutateData {
	// a dataset named `array` is now expected to already exist
	array: ndarray::Array3<usize>,

	// a dataset `create_me_dataset` will be created 
	#[hdf5(mutate_on_write=false)]
	create_me_dataset: ndarray::Array3<usize>
}
```

### Mutating with different shaped data

If you are reading in some data, mutating the shape in any way (including `#[transpose="write"]` / `#[transpose="read"]`),
and then writing it to the same file `hdf5` will throw an error. Because `hdf5` has no mechanism 
to delete data from a group, you will have to create a new [`File`] object and write all of the data there. 

If the data you are operating on never changes shape (or you use `#[transpose="both"]`), this will not be an issue. Alternatively,
you can also avoid the issue if you dont use `#[mutate_on_write]` and instead write to a new file.

## Attributes

You can also store scalar-valued attributes along with array.
All the previous attributes (with the exception of transposing) can be applied to scalar attributes as
well.

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::Array3;
use ndarray::Array5;

#[derive(ContainerRead, ContainerWrite)]
struct SolverResultWithAttribute {
	high_dimensional_data: Array5<f64>,
	#[hdf5(mutate_on_write)]
	current_timestep: u32,
}
```

## HDF5 Groups

you can nest `struct` definitions with derived `ContainerRead` / `ContainerWrite` in order to 
parse nested groups. For example if you had the following layout:

```text
.
├── attribute_1
├── attribute_2
├── group_1
│   ├── dataset_1
│   ├── dataset_2
│   └── dataset_3
└── group_2
    ├── attribute_1
    ├── dataset_1
    └── dataset_2
```

you could parse them with

```rust
use hdf5_derive::{ContainerRead, ContainerWrite};
use ndarray::{Array3, Array2};

#[derive(ContainerRead, ContainerWrite)]
struct Root {
	attribute_1: usize,
	attribute_2: u16,
	#[hdf5(transpose="both")] // <---- this attribute does nothing
	group_1: FirstGroup,
	#[hdf5(mutate_on_write)] // <---- this attribute does nothing
	group_2: SecondGroup,
}

#[derive(ContainerRead, ContainerWrite)]
#[hdf5(mutate_on_write)] // <-- this attribute works
struct FirstGroup {
	dataset_1: Array3<u8>,
	dataset_2: Array2<f32>,
	dataset_3: Array2<usize>,
}

#[derive(ContainerRead, ContainerWrite)]
#[hdf5(transpose= "read")] // <-- this attribute works
struct SecondGroup {
	dataset_1: Array3<f64>,
	dataset_2: Array2<i32>,
}
```
