package Drago

import (
	"math"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

func matrixWithInitialValue(r, c int, val float64) *mat64.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = val
	}
	return mat64.NewDense(r, c, data)
}

func randomMatrix(r, c int) *mat64.Dense {
	rand.Seed(time.Now().UnixNano())
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.Float64() / 2.0
	}
	return mat64.NewDense(r, c, data)
}

func normalize(input []float64) []float64 {
	total := 0.0
	for _, val := range input {
		total += val * val
	}
	total = math.Sqrt(total)
	res := make([]float64, len(input))
	for i, val := range input {
		res[i] = val / total
	}
	return res
}
