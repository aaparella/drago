package Drago

import "math"

func MSE(predictions, actual []float64) float64 {
	sum := 0.0
	for i := range predictions {
		sum += math.Pow(predictions[i]-actual[i], 2.0)
	}
	return sum / float64(len(predictions))
}
