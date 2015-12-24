package main

import "github.com/aaparella/Drago"

func main() {
	activations := []Drago.Activator{new(Drago.Sigmoid), new(Drago.Sigmoid)}
	net := Drago.New(0.1, 25, []int{5, 2, 3, 1}, activations)
	net.Learn([][][]float64{
		{{.1, .2, .3, .4, .5}, {1}},
		{{.2, .5, .3, .7, .8}, {2}},
		{{.2, .5, .3, .7, .8}, {2}},
		{{.2, .5, .3, .7, .8}, {2}},
		{{.2, .5, .3, .7, .8}, {2}},
		{{.2, .5, .3, .7, .8}, {2}},
	})
}
