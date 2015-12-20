package Drago

import (
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

func MatrixWithInitialValue(r, c int, val float64) *mat64.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = val
	}
	return mat64.NewDense(r, c, data)
}

func RandomMatrix(r, c int) *mat64.Dense {
	rand.Seed(time.Now().UnixNano())
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.Float64()
	}
	return mat64.NewDense(r, c, data)
}
