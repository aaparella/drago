package Drago

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type MSE struct {
}

var _ Criterion = new(MSE)

func (m *MSE) Derivative(prediction, actual *mat64.Dense) *mat64.Dense {
	mat := &mat64.Dense{}
	mat.Sub(prediction, actual)
	return mat
}

func (m *MSE) Apply(prediction, actual *mat64.Dense) float64 {
	rows, _ := prediction.Dims()
	error := 0.0
	for i := 0; i < rows; i++ {
		diff := prediction.At(i, 0) - actual.At(i, 0)
		error += math.Pow(diff, 2)
	}
	return error / float64(rows)
}
