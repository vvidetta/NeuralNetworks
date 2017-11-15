using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public delegate double BinaryOperation(double x, double y);

    public class Matrix
    {
        #region Data Members

        private double[,] matrix_;
        private int height_;
        private int width_;

        #endregion

        #region Construction

        public Matrix(int height, int width)
        {
            this.height_ = height;
            this.width_ = width;
            this.matrix_ = new double[height, width];
        }

        public Matrix(double [,] matrix)
        {
            this.matrix_ = matrix;
            this.height_ = matrix.GetLength(0);
            this.width_ = matrix.GetLength(1);
        }

        /// <summary>
        /// Copy Constructor
        /// </summary>
        /// <param name="matrix">The Matrix to copy</param>
        public Matrix(Matrix matrix) : this(matrix.matrix_) { }

        #endregion

        #region Operations

        public static implicit operator Matrix(double[,] array)
        {
            return new Matrix(array); 
        }

        public static implicit operator Matrix(double[] vector)
        {
            var colVector = new double[vector.Length, 1];
            for (int i = 0; i < vector.Length; i++)
            {
                colVector[i, 0] = vector[i];
            }

            return new Matrix(colVector);
        }

        public static Matrix operator*(double c, Matrix A)
        {
            RealFunc multiply = (x) => { return c * x; };
            return Matrix.ApplyFunction(multiply, A);
        }

        public static Matrix operator* (Matrix A, Matrix B)
        {
            if (A.Width != B.Height)
            {
                throw new Exception("Dimension mismatch !!!!!");
            }

            Matrix C = new Matrix(A.Height, B.Width);

            for (int i = 0; i < A.Height; i++)
            {
                for (int j = 0; j< B.Width; j++)
                {
                    C[i, j] = 0;
                    for (int k = 0; k < A.Width; k++)
                    {
                        C[i, j] += A[i, k] * B[k, j];
                    }
                }
            }

            return C;
        }

        public static Matrix ApplyOperation(BinaryOperation op, Matrix A, Matrix B)
        {
            if (A.Height != B.Height || A.Width != B.Width)
                throw new Exception("Dimension mismatch!!!!!!!!!!!!!!!!!!!!!");

            Matrix C = new Matrix(A.Height, A.Width);

            for (int i = 0; i < C.Height; i++)
            {
                for (int j = 0; j < C.Width; j++)
                {
                    C[i, j] = op(A[i, j], B[i, j]);
                }
            }

            return C;
        }

        public static Matrix HadamardProduct(Matrix A, Matrix B)
        {
            BinaryOperation times = (x, y) => { return x * y; };
            return ApplyOperation(times, A, B);
        }

        public static Matrix operator+(Matrix A, Matrix B)
        {
            BinaryOperation add = (x, y) => { return x + y; };
            return ApplyOperation(add, A, B);
        }

        public static Matrix operator-(Matrix A, Matrix B)
        {
            BinaryOperation minus = (x, y) => { return x - y; };
            return ApplyOperation(minus, A, B);
        }

        public static Matrix Transpose(Matrix A)
        {
            Matrix B = new Matrix(A.Width, A.Height);

            for (int i = 0; i < B.Height; i++)
                for (int j = 0; j < B.Width; j++)
                    B[i, j] = A[j, i];

            return B;
        }

        public static Matrix ApplyFunction(RealFunc func, Matrix A)
        {
            Matrix B = new Matrix(A.Height, A.Width);

            for (int i = 0; i < A.Height; i++)
            {
                for (int j = 0; j < A.Width; j++)
                {
                    B[i, j] = func(A[i, j]);
                }
            }

            return B;
        }

        public static Matrix CreateRandom(int height, int width)
        {
            Matrix A = new Matrix(height, width);
            Random rand = new Random();
            RealFunc f = (x) => { return rand.NextDouble(); };
            return ApplyFunction(f, A);
        }

        public double Error(Matrix Expected)
        {
            if (Expected.Height != Height || Expected.Width != Width)
                throw new Exception("Dimension mismatch!!!!!");

            double acc = 0;
            double e = 0;
            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                {
                    e = this[i, j] - Expected[i, j];
                    acc += e * e;
                }

            acc /= 2;
            return acc;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder("[");
            for (int i = 0; i < height_; i++)
            {
                builder.Append("[ ");
                for (int j = 0; j < width_; j++)
                {
                    if (j != 0)
                        builder.Append(", ");

                    builder.Append(this[i, j]);
                }
                builder.Append("]");
            }
            builder.Append("]");

            return builder.ToString();
        }

        #endregion

        #region Fields

        public double this[int i, int j]
        {
            get { return this.matrix_[i, j]; }
            set { this.matrix_[i, j] = value; }
        }

        public int Height
        {
            get { return this.height_; }
        }

        public int Width
        {
            get
            {
                return this.width_;
            }
        }

        #endregion
    }
}
