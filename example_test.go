package drago_test

import (
	"fmt"

	"github.com/aaparella/drago"
)

func ExamplePredict() {
	activations := []drago.Activator{new(drago.Sigmoid)}
	net := drago.New(.7, 1000, []int{2, 3, 1}, activations)
	net.Learn([][][]float64{
		{{0, 0}, {0}},
		{{1, 0}, {0}},
		{{0, 1}, {0}},
		{{1, 1}, {1}},
	})
	fmt.Println(net.Predict([]float64{0, 0}))
	fmt.Println(net.Predict([]float64{0, 1}))
	fmt.Println(net.Predict([]float64{1, 0}))
	fmt.Println(net.Predict([]float64{1, 1}))
}
