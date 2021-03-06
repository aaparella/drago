// Example borrowed from https://github.com/stevenmiller888/go-mind

package drago_test

import (
	"log"

	"github.com/aaparella/drago"
)

var (
	a = character(
		".#####." +
			"#.....#" +
			"#.....#" +
			"#######" +
			"#.....#" +
			"#.....#" +
			"#.....#")
	b = character(
		"######." +
			"#.....#" +
			"#.....#" +
			"######." +
			"#.....#" +
			"#.....#" +
			"######.")
	c = character(
		"#######" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#######")
)

func ExampleLearn() {
	m := drago.New(0.3, 10000, []int{49, 3, 1}, []drago.Activator{new(drago.Sigmoid)})
	m.Learn([][][]float64{
		{c, []float64{.5}},
		{b, []float64{.3}},
		{a, []float64{.1}},
	})

	result := m.Predict(
		character(
			"#######" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"#######"))
	log.Println(result)
}
func character(chars string) []float64 {
	flt := make([]float64, len(chars))
	for i := 0; i < len(chars); i++ {
		if chars[i] == '#' {
			flt[i] = 1.0
		} else { // if '.'
			flt[i] = 0.0
		}
	}
	return flt
}
