package Drago

import "github.com/gonum/matrix/mat64"

type Error func([]float64, []float64) float64

func ApplyFunc(fun func(int, int, float64) float64, mat *mat64.Dense) *mat64.Dense {
	res := mat64.DenseCopyOf(mat)
	res.Apply(fun, res)
	return res
}
