package main

import (
	"fmt"

	"github.com/aaparella/Drago"
)

func main() {
	net := Drago.New(.1, 1000, []int{4, 3, 1}, []Drago.Activator{new(Drago.Sigmoid)})
	// Note : This data set is entirely made up, just demonstrative
	// In each sample the features are, in order :
	// Number of seats in theatre
	// Ticket price
	// Band's followers on Facebook
	// Days since band has had last concert
	//
	// The label being predicted is how many people show up
	samples := [][][]float64{
		{{100, 10, 1000, 30}, {90}},
		{{50, 20, 500, 10}, {25}},
		{{100, 50, 1000, 15}, {30}},
		{{75, 10, 1000, 100}, {70}},
	}
	net.Learn(samples)
	fmt.Println(net.Predict([]float64{40, 5, 400, 17}))
}
