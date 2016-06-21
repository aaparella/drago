package main

import (
	"fmt"

	"github.com/aaparella/drago"
)

func main() {
	acts := []drago.Activator{new(drago.Sigmoid)}
	net := drago.New(.7, 1000, []int{2, 3, 1}, acts)
	net.Learn([][][]float64{
		{{0, 0}, {0}},
		{{1, 0}, {1}},
		{{0, 1}, {1}},
		{{1, 1}, {0}},
	})

	fmt.Println("XOR value for {0, 0}, =>", net.Predict([]float64{0, 0}))
	fmt.Println("XOR value for {1, 0}, =>", net.Predict([]float64{1, 0}))
	fmt.Println("XOR value for {0, 1}, =>", net.Predict([]float64{0, 1}))
	fmt.Println("XOR value for {1, 1}, =>", net.Predict([]float64{1, 1}))
}
