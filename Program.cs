using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public delegate double LossFunction(List<OutputNode> outputs);

    class Program
    {
        public static double sechSquared(double x)
        {
            double cosh = Math.Cosh(x);
            return 1 / (cosh * cosh);
        }

        public static double sigmoidAntiPrime(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public static double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double sigmoidPrime(double x)
        {
            double s = sigmoid(x);
            return s * (1 - s);
        }
        
        public static double ReLU(double x)
        {
            return (x > 0.0) ? x : 0.0;
        }

        public static double ReLUPrime(double x)
        {
            return (x > 0.0) ? 1.0 : 0.0;
        }

        static void Main(string[] args)
        {
            MatrixNeuralNetwork net = MatrixNeuralNetwork.CreateRandom(3, 1, Math.Tanh); // sigmoid);

            TrainingSet trainingData = new TrainingSet();
            //trainingData.Input = (Math.PI) * Matrix.CreateRandom(50, 1);
            //trainingData.Output = Matrix.ApplyFunction(Math.Sin, trainingData.Input);

            trainingData.Input = RandomArithmetic(100);
            trainingData.Output = EvaluateArithmetic(trainingData.Input);

            Console.WriteLine("Input: {0}", trainingData.Input);
            Console.WriteLine("Expected: {0}", trainingData.Output);
            Console.WriteLine("Press any key to begin optimisation...");
            Console.ReadKey();

            MNNOptimiserOptions options = new MNNOptimiserOptions();
            options.actFuncPrime = sechSquared; //sigmoid; // ReLUPrime; // sigmoidPrime;
            options.errorBound = 1e-3;
            options.learningRate = 1e-6;

            MNNOptimiser optimiser = new MNNOptimiser(SquareErrorLoss);
            optimiser.Optimise(net, trainingData, options);

            net.Evaluate(trainingData.Input);
            Matrix observed = net.Output;
            Matrix error = observed - trainingData.Output;
            Console.WriteLine();
            Console.WriteLine("Errors = {0}", error);

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
        
        public static double SquareErrorLoss(MatrixNeuralNetwork net, TrainingSet trainingData)
        {
            net.Evaluate(trainingData.Input);
            Matrix errors = net.Output - trainingData.Output;
            errors = Matrix.HadamardProduct(errors, errors);
            double acc = 0;
            for (int i = 0; i < errors.Height; i++)
                for (int j = 0; j < errors.Width; j++)
                    acc += errors[i, j];

            return acc * 0.5;
        }

        public static RealFunc D(RealFunc f)
        {
            double delta = Math.Pow(2, -26);
            RealFunc df = (x) => { return (f(x + delta) - f(x - delta)) / (2 * delta); };
            return df;
        }

        public static int[] opcodes = { '+', '-', '*', '/' };

        public static Matrix RandomArithmetic(int height)
        {
            Matrix result = new Matrix(height, 3);
            Random rand = new Random();
            for (int i = 0; i < height; i++)
            {
                result[i, 0] = rand.Next();
                result[i, 1] = opcodes[rand.Next(4)];
                result[i, 2] = rand.Next();
            }

            return result;
        }

        public static Matrix EvaluateArithmetic(Matrix Input)
        {
            if (Input.Width != 3)
                throw new Exception("Invalid input!!!!!!!");

            Matrix Result = new Matrix(Input.Height, 1);
            for (int i = 0; i < Result.Height; i++)
            {
                switch ((int)Input[i, 1])
                {
                    case '+':
                        Result[i, 0] = Input[i, 0] + Input[i, 2];
                        break;

                    case '-':
                        Result[i, 0] = Input[i, 0] - Input[i, 2];
                        break;

                    case '*':
                        Result[i, 0] = Input[i, 0] * Input[i, 2];
                        break;

                    case '/':
                        Result[i, 0] = Input[i, 0] / Input[i, 2];
                        break;

                    default:
                        throw new Exception("Unknown Operation!!!!!");
                }
            }

            return Matrix.ApplyFunction(sigmoid, Result);
        }
    }
}
