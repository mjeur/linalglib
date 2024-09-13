use linalg::Matrix;
use linalg::F64type;



fn main() {
    println!(">>>");
    let matrix1 = 
        Matrix::<F64type>::from_file("src/m1_3x3.txt")
        .expect("Failed to read matrix from file");
    let matrix2 = Matrix::<F64type>::from_file("src/m2_3x3.txt").expect("Failed to read matrix from file");
    let mul_result = matrix1.mul(&matrix2);
    let sum_result = matrix1.add(&matrix2);
    mul_result.show();
    sum_result.show();
}