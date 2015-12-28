package main

import (
	"fmt"

	"github.com/aaparella/Drago"
)

func main() {
	activations := []Drago.Activator{new(Drago.Sigmoid)}
	net := Drago.New(.5, 50, []int{2, 3, 1}, activations)
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
