use::std::fmt;
use std::fmt::Debug;
use std::fs;
use std::io;
use std::str::FromStr;
use std::ops::{Add, Mul, Sub, Neg, Div};

pub struct Matrix<T> {
    pub rows:usize,
    pub cols:usize,
    pub data: Vec<Vec<T>>,
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}
pub trait One {
    fn one() -> Self;
}

#[derive(Debug, Clone, Default)]
pub struct F64type {
    value: f64,
}

impl FromStr for F64type {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = s.parse::<f64>().map_err(|_| "Invalid value".to_string())?;
        Ok(F64type { value })
    }
}

impl Add for F64type {
    type Output = F64type;

    fn add(self, other: F64type) -> Self::Output {
        F64type {
            value: self.value + other.value,
        }
    }
}
impl Sub for F64type {
    type Output = F64type;

    fn sub(self, other: F64type) -> Self::Output {
        F64type {
            value: self.value - other.value,
        }
    }
}

impl Mul for F64type {
    type Output = F64type;

    fn mul(self, other: F64type) -> Self::Output {
        F64type {
            value: self.value * other.value,
        }
    }
}

impl Div for F64type {
    type Output = F64type;

    fn div(self, other: F64type) -> Self::Output {
        F64type {
            value: self.value / other.value,
        }
    }
}

impl One for F64type {
    fn one() -> Self {
        F64type { value: 1.0 }
    }
}

impl Neg for F64type {
    type Output = F64type;

    fn neg(self) -> Self::Output {
        F64type { value: -self.value }
    }
}

impl PartialEq for F64type {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Sqrt for F64type {
    fn sqrt(self) -> Self {
        F64type { value: self.value.sqrt() }
    }
}

impl fmt::Display for F64type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T> Matrix<T> 
where T: Default + Clone + FromStr<Err = String> + 
Debug + fmt::Display + std::ops::Mul<Output = T> + 
std::ops::Add<Output = T> + std::ops::Sub<Output = T> +
std::cmp::PartialEq + One + std::ops::Neg<Output = T> +
std::ops::Div<Output = T> + Sqrt
{
    
    pub fn new(rows:usize, cols:usize) -> Self {
        let data = vec![vec![T::default(); cols]; rows];
        return Matrix{ rows, cols, data};
    }

    pub fn from_file(path: &str) ->  Result<Self, io::Error> {
        let file_data = fs::read_to_string(path).unwrap_or_else(|e| panic!("Error while reading file: {e}"));
        let mut matrix: Vec<Vec<T>> = Vec::new();

        for rows in file_data.lines() {
            let mut row: Vec<T> = Vec::new();
            let entries: Vec<&str> = rows.split_whitespace().collect();

            for val in entries {
                match val.parse::<T>() {
                    Ok(value) => row.push(value),
                    Err(err) => return Err(io::Error::new(io::ErrorKind::InvalidData, err.to_string())),
                }
            }

            matrix.push(row);
        }

        let r = matrix.len();
        let c = matrix[0].len();
        return Ok(Matrix {rows: r, cols: c, data: matrix});
    }

    pub fn get(&self, row:usize, col:usize) -> T {
        self.data[row][col].clone()
    }

    pub fn set(&mut self, row:usize, col:usize, value:T) {
        self.data[row][col] = value
    }


    pub fn show(&self) {
        for i in 0..self.rows {
            println!();
            for j in 0..self.cols {
                print!("{} ", self.get(i, j))
            }
        }
    }

    pub fn add(&self, other:&Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i,j, self.get(i, j) + other.get(i, j));
            }
        }
        result
    }

    // pub fn mul(&self, other: &Matrix<T>) -> Matrix<T> {
    //     assert!(self.cols == other.rows);

    //     let mut result = Matrix::new(self.rows, other.cols);
    //     for i in 0..self.rows {
    //         for j in 0..other.cols {
    //             let mut sum = T::default();
    //             for k in 0..self.cols {
    //                 sum = sum + (self.get(i, k) * other.get(k, j));
    //             }
    //             result.data[i][j] = sum;
    //         }
    //     }
    //     result
    // }


    pub fn mul(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn sub(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) - other.get(i, j));
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix<T> {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn minor(&self, row: usize, col: usize) -> Matrix<T> {
        let mut result = Matrix::new(self.rows - 1, self.cols - 1);
        for i in 0..self.rows {
            if i == row {
                continue;
            }
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                let new_i = if i < row { i } else { i - 1 };
                let new_j = if j < col { j } else { j - 1 };
                result.set(new_i, new_j, self.get(i, j));
            }
        }
        result
    }

    pub fn det(&self) -> T {
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }
        if self.rows == 1 {
            self.get(0, 0)
        } else if self.rows == 2 {
            self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0)
        } else {
            let mut result = T::default();
            for i in 0..self.cols {
                let minor = self.minor(0, i);
                result = result + self.get(0, i) * minor.det() * if i % 2 == 0 { T::one() } else { -T::one() };
            }
            result
        }
    }

    pub fn rank(&self) -> usize {
        let mut result = 0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if self.get(i, j) != T::default() {
                    result += 1;
                }
            }
        }
        result
    }

    pub fn inv(&self) -> Matrix<T> {
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }
        let det = self.det();
        if det == T::default() {
            panic!("Matrix is not invertible");
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let det_clone = det.clone();
                let minor = self.minor(i, j);
                result.set(i, j, minor.det() * if (i + j) % 2 == 0 { T::one() } else { -T::one() } / det_clone);
            }
        }
        result
    }

    pub fn neg(&self) -> Matrix<T> {
        let mut matrix = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix.set(i, j, T::default() - self.get(i, j));
            }
        }
        matrix
    }

    pub fn trace(&self) -> T {
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }
        let mut result = T::default();
        for i in 0..self.rows {
            result = result + self.get(i, i);
        }
        result
    }

    pub fn eigenvalue(&self, index: usize) -> T {
        let mut v = vec![T::default(); self.cols];
        v[index] = T::one();
    
        for _ in 0..100 {
            let v_mat = Matrix { rows: self.cols, cols: 1, data: v.iter().map(|x| vec![x.clone()]).collect() };
            let v_new_mat = self.mul(&v_mat);
    
            let norm = v_new_mat.norm();
            for i in 0..self.cols {
                v[i] = v_new_mat.get(i, 0) / norm.clone();
            }
        }
    
        let v_mat = Matrix { rows: self.cols, cols: 1, data: v.iter().map(|x| vec![x.clone()]).collect() };
        let v_new_mat = self.mul(&v_mat);
    
        v_new_mat.get(0, 0)
    }

    pub fn map<U>(&self, f: impl Fn(T) -> U) -> Matrix<U>
    where
        U: Default + Clone + FromStr<Err = String> + Debug + std::fmt::Display + std::ops::Add<Output = U> + std::ops::Sub<Output = U> + std::ops::Mul<Output = U> + std::ops::Div<Output = U> + std::ops::Neg<Output = U> + PartialEq + One + Sqrt,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, f(self.get(i, j)));
            }
        }
        result
    }

    pub fn norm(&self) -> T {
        let mut sum = T::default();
        for i in 0..self.rows {
            for j in 0..self.cols {
                sum = sum + self.get(i, j) * self.get(i, j);
            }
        }
        sum.sqrt()
    }


    pub fn eigenvector(&self, index: usize) -> Matrix<T> {
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }
        let eigenvalue = self.eigenvalue(index);
        let mut result = Matrix::new(self.rows, 1);
        for i in 0..self.rows {
            result.set(i, 0, self.get(i, index) / eigenvalue.clone());
        }
        result
    }

    pub fn eigenvalues(&self) -> Vec<T> {
        if self.rows != self.cols {
            panic!("Matrix is not square");
        }
        let mut result = Vec::new();
        for i in 0..self.rows {
            let eigenvalue = self.eigenvalue(i);
            result.push(eigenvalue);
        }
        result
    }

}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
