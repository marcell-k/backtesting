use pyo3::prelude::*;

#[pymodule]
mod _native {
    use pyo3::prelude::*;

    #[pyfunction]
    fn ping() -> PyResult<&'static str> {
        Ok("pong")
    }
}
