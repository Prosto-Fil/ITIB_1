#include <iostream>
#include <vector>
#include <cmath>
#include <bitset>

double learning_rate = 0.3;
std::vector<std::vector<int>> dec = {
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 1, 0},
        {1, 0, 0, 1, 1},
        {1, 0, 1, 0, 0},
        {1, 0, 1, 0, 1},
        {1, 0, 1, 1, 0},
        {1, 0, 1, 1, 1},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 1},
        {1, 1, 0, 1, 0},
        {1, 1, 0, 1, 1},
        {1, 1, 1, 0, 0},
        {1, 1, 1, 0, 1},
        {1, 1, 1, 1, 0},
        {1, 1, 1, 1, 1}
};

int activation_func_1 (double net) {
    if (net < 0) {
        return 0;
    } else {
        return 1;
    }
}

double activation_func_2 (double net) {
    double res = 0.5 * (tanh(net) + 1);
    return round(res * 100) / 100;
}

void neuron_system_first_second(std::vector<int>& function, uint activation_func_num) {
    std::vector<double> weights = {0, 0, 0, 0, 0};
    std::vector<double> deltas = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int epoch_counter = 0;
    int num_of_mistakes = 1;
    while(num_of_mistakes != 0) {
        num_of_mistakes--;
        std::cout << "Epoch " << epoch_counter << ":" << std::endl;

        std::cout << "Weights: ";
        for (double weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << std::endl;

        std::vector<double> y_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        num_of_mistakes = 0;

        std::vector<double> previous_weights = weights;
        for (int i = 0; i < function.size(); i++) {
            double net = weights[0] +
                         dec[i][1] * weights[1] +
                         dec[i][2] * weights[2] +
                         dec[i][3] * weights[3] +
                         dec[i][4] * weights[4];

            net = round(net * 100) / 100;

            double previous_net = previous_weights[0] +
                                  dec[i][1] * previous_weights[1] +
                                  dec[i][2] * previous_weights[2] +
                                  dec[i][3] * previous_weights[3] +
                                  dec[i][4] * previous_weights[4];

            double f_net;
            if (activation_func_num == 1) {
                f_net = activation_func_1(net);
                y_vec[i] = activation_func_1(previous_net);
            } else if (activation_func_num == 2) {
                f_net = activation_func_2(net);

                if (activation_func_2(previous_net) >= 0.5) {
                    y_vec[i] = 1;
                } else {
                    y_vec[i] = 0;
                }
            } else {
                exit(0);
            }

            double delta = double(function[i]) - f_net;
            delta = round(delta * 100) / 100;
            deltas[i] = delta;

            if (delta != 0) {
                double derivative;
                if (activation_func_num == 1) {
                    derivative = 1;
                } else if (activation_func_num == 2) {
                    derivative = (1 / (2 * pow(cosh(net), 2)));
                    derivative = round(derivative * 100) / 100;
                } else {
                    exit(0);
                }

                weights[0] = weights[0] + learning_rate * delta * derivative;
                weights[0] = round(weights[0] * 100) / 100;

                weights[1] = weights[1] + learning_rate * delta * double(dec[i][1]) * derivative;
                weights[1] = round(weights[1] * 100) / 100;

                weights[2] = weights[2] + learning_rate * delta * double(dec[i][2]) * derivative;
                weights[2] = round(weights[2] * 100) / 100;

                weights[3] = weights[3] + learning_rate * delta * double(dec[i][3]) * derivative;
                weights[3] = round(weights[3] * 100) / 100;

                weights[4] = weights[4] + learning_rate * delta * double(dec[i][4]) * derivative;
                weights[4] = round(weights[4] * 100) / 100;
            }
        }

        std::cout << "y: ";
        for (double y : y_vec) {
            std::cout << y;
        }
        std::cout << std::endl;

        for (int i = 0; i < y_vec.size(); i++) {
            if (y_vec[i] != function[i]) {
                num_of_mistakes++;
            }
        }

        std::cout << "Number of mistakes: ";
        std::cout << num_of_mistakes << std::endl << std::endl;

        epoch_counter++;
    }
}

void neuron_system_third(std::vector<int>& function) {
    int current_vectors_size = 15;

    std::vector<std::pair<std::bitset<16>, int>> combinations;
    combinations.reserve(65536);
    for (int i = 0; i < 65536; i++) {
        std::bitset<16> bits(i);
        int current_weight = 0;
        for (int j = 0; j < 16; j++) {
            if (bits[j] == 1) {
                current_weight++;
            }
        }
        combinations.emplace_back(bits, current_weight);
    }

    std::vector<std::vector<int>> last_working_combination = dec;
    std::vector<double> last_working_weights;
    int last_epoch;
    for (auto& combination : combinations) {
        if (combination.second == current_vectors_size) {
            std::vector<std::bitset<16>> current_variants;
            current_variants.push_back(combination.first);

            std::vector<std::vector<int>> current_dec;
            bool working = 0;

            for (auto& current_variant : current_variants) {
                working = 0;
                current_dec.reserve(current_vectors_size);

                for (int j = 0; j < dec.size(); j++) {
                    if (current_variant[j] == 1) {
                        current_dec.push_back(dec[j]);
                    }
                }

                std::vector<double> weights = {0, 0, 0, 0, 0};
                std::vector<double> deltas;
                deltas.reserve(current_vectors_size);
                for(int j = 0; j < current_vectors_size; j++) {
                    deltas.push_back(0);
                }

                int epoch_counter = 0;
                int num_of_mistakes = 1;
                while(num_of_mistakes != 0 && epoch_counter < 40) {
                    num_of_mistakes--;

                    std::vector<double> y_vec;
                    y_vec.reserve(current_vectors_size);
                    for(int j = 0; j < current_vectors_size; j++) {
                        y_vec.push_back(0);
                    }

                    num_of_mistakes = 0;

                    std::vector<double> previous_weights = weights;
                    for (int i = 0; i < function.size(); i++) {
                        double net = weights[0] +
                                     current_dec[i][1] * weights[1] +
                                     current_dec[i][2] * weights[2] +
                                     current_dec[i][3] * weights[3] +
                                     current_dec[i][4] * weights[4];

                        net = round(net * 100) / 100;

                        double previous_net = previous_weights[0] +
                                              current_dec[i][1] * previous_weights[1] +
                                              current_dec[i][2] * previous_weights[2] +
                                              current_dec[i][3] * previous_weights[3] +
                                              current_dec[i][4] * previous_weights[4];

                        double f_net = activation_func_2(net);
                        if (activation_func_2(previous_net) >= 0.5) {
                            y_vec[i] = 1;
                        } else {
                            y_vec[i] = 0;
                        }

                        double delta = double(function[i]) - f_net;
                        delta = round(delta * 100) / 100;
                        deltas[i] = delta;

                        if (delta != 0) {
                            double derivative = (1 / (2 * pow(cosh(net), 2)));
                            derivative = round(derivative * 100) / 100;

                            weights[0] = weights[0] + learning_rate * delta * derivative;
                            weights[0] = round(weights[0] * 100) / 100;

                            weights[1] = weights[1] + learning_rate * delta * double(current_dec[i][1]) * derivative;
                            weights[1] = round(weights[1] * 100) / 100;

                            weights[2] = weights[2] + learning_rate * delta * double(current_dec[i][2]) * derivative;
                            weights[2] = round(weights[2] * 100) / 100;

                            weights[3] = weights[3] + learning_rate * delta * double(current_dec[i][3]) * derivative;
                            weights[3] = round(weights[3] * 100) / 100;

                            weights[4] = weights[4] + learning_rate * delta * double(current_dec[i][4]) * derivative;
                            weights[4] = round(weights[4] * 100) / 100;
                        }
                    }

                    for (int i = 0; i < y_vec.size(); i++) {
                        if (y_vec[i] != function[i]) {
                            num_of_mistakes++;
                        }
                    }

                    epoch_counter++;
                }

                if (num_of_mistakes != 0) {
                    working = 0;
                } else {
                    std::vector<double> y_vec;
                    for (int i = 0; i < function.size(); i++) {
                        double net = weights[0] +
                                     dec[i][1] * weights[1] +
                                     dec[i][2] * weights[2] +
                                     dec[i][3] * weights[3] +
                                     dec[i][4] * weights[4];

                        net = round(net * 100) / 100;

                        y_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                        if (activation_func_2(net) >= 0.5) {
                            y_vec[i] = 1;
                        } else {
                            y_vec[i] = 0;
                        }
                    }

                    int new_num_of_mistakes = 0;
                    for (int i = 0; i < y_vec.size(); i++) {
                        if (y_vec[i] != function[i]) {
                            new_num_of_mistakes++;
                        }
                    }

                    if (new_num_of_mistakes == 0) {
                        working = 1;
                        last_working_weights = weights;
                    } else {
                        working = 0;
                    }
                }
                last_epoch = epoch_counter;
            }

            if (working == 1) {
                last_working_combination = current_dec;
                current_vectors_size--;
            } else {
                break;
            }
        }
    }

    std::cout << "Subsets: ";
    for (int i = 0; i < last_working_combination.size(); i++) {
        for (int j = 0; j < dec.size(); i++) {
            if (last_working_combination[i] == dec[j]) {
                std::cout << j << " ";
            }
        }
    }
    std::cout << std::endl;

    std::cout << "Weights: ";
    for (double weight : last_working_weights) {
        std::cout << weight << " ";
    }
    std::cout << std::endl;

    std::cout << "Epoch: " << last_epoch << std::endl;
}

int main() {
    std::vector<int> function;

    // x1 + !x2 + !(x3+x4)
    function.reserve(16);
    std::cout << "Function: ";
    for (int i = 0; i < 16; i++) {
        function.push_back(dec[i][1] || !dec[i][2] || !(dec[i][3] || dec[i][4]));
        std::cout << function[i];
    }
    std::cout << std::endl << std::endl;

    std::cout << "FIRST TASK:" << std::endl;
    neuron_system_first_second(function, 1);

    std::cout << "SECOND TASK:" << std::endl;
    neuron_system_first_second(function, 2);

    std::cout << "THIRD TASK:" << std::endl;
    neuron_system_third(function);
    return 0;
}
