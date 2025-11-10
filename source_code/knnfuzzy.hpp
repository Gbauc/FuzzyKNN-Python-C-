#ifndef FUZZY_KNN_HPP
#define FUZZY_KNN_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include <unordered_map>

template <typename T>
std::vector<int> argsort(const std::vector<T>& array) {
    std::vector<int> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int i1, int i2) { return array[i1] < array[i2]; });
    return indices;
}

template <typename T>
struct KNNResult {
    std::vector<int> predictions;
    std::vector<std::unordered_map<int, double>> memberships;
};

template <typename T>
class KNN {
public:
    int k;
    double m;

    KNN(int k_neighbors, double fuzziness = 2.0)
        : k(k_neighbors), m(fuzziness) {}

    void _fit(const std::vector<std::vector<T>>& x_train, const std::vector<int>& y_train);

    KNNResult<T> _predict(const std::vector<std::vector<T>>& x_test);


private:
    std::vector<std::vector<T>> stored_x_train;
    std::vector<int> stored_y_train;
    std::unordered_map<int, std::vector<double>> stored_membership;
    std::unordered_map<int, int> count_classes(const std::vector<int>& K_neighbors, const std::vector<int>& y_train);
    std::unordered_map<int, std::vector<double>> fuzzy_membership(const std::vector<std::vector<T>>& train_matrix, const std::vector<int>& labels);
    std::vector<int> k_near(int k, const std::vector<std::vector<T>>& source_matrix, const std::vector<T>& target_vector);
    T calculate_distance(const std::vector<T>& vector1,  const std::vector<T>& vector2);
};

template <typename T>
std::unordered_map<int, int> KNN<T>::count_classes(const std::vector<int>& K_neighbors, const std::vector<int>& y_train) {
    std::unordered_map<int, int> counts;
    for (int index : K_neighbors) {
        int label = y_train[index];
        counts[label]++;
    }
    return counts;
}

template <typename T>
std::unordered_map<int, std::vector<double>> KNN<T>::fuzzy_membership(
    const std::vector<std::vector<T>>& train_matrix,
    const std::vector<int>& labels) {

    std::vector<int> classes = labels;
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());

    std::unordered_map<int, std::vector<double>> membership_classes;
    for (int c : classes)
        membership_classes[c] = std::vector<double>(labels.size(), 0.0);

    for (size_t i = 0; i < labels.size(); i++) {
        auto X = train_matrix[i];
        int y = labels[i];

        auto X_neighbors = k_near(k, train_matrix, X);
        auto counts = count_classes(X_neighbors, labels);

        for (int c : classes) {
            double uc = 0.0;
            if (counts.find(c) != counts.end()) {
                uc = 0.49 * (static_cast<double>(counts[c]) / k);
                if (c == y)
                    uc += 0.51;
            }
            membership_classes[c][i] = uc;
        }
    }

    return membership_classes;
}

template <typename T>
T KNN<T>::calculate_distance(const std::vector<T>& vector1, const std::vector<T>& vector2) {
    T distance = 0.0;
    for (size_t i = 0; i < vector1.size(); i++) {
        distance += std::pow(vector1[i] - vector2[i], 2);
    }
    return std::sqrt(distance);
}

template <typename T>
void KNN<T>::_fit(const std::vector<std::vector<T>>& x_train,  const std::vector<int>& y_train) {
    this->stored_x_train = x_train;
    this->stored_y_train = y_train;
    this->stored_membership = fuzzy_membership(this->stored_x_train, this->stored_y_train);
}

template <typename T>
std::vector<int> KNN<T>::k_near(int k, const std::vector<std::vector<T>>& source_matrix, const std::vector<T>& target_vector) {
    std::vector<T> distances;
    for (const auto& train_vector : source_matrix) {
        distances.push_back(calculate_distance(target_vector, train_vector));
    }

    std::vector<int> indexes = argsort(distances);
    std::vector<int> neighbors_index;
    for (int i = 0; i < k; i++)
        neighbors_index.push_back(indexes[i]);

    return neighbors_index;
}


template <typename T>
KNNResult<T> KNN<T>::_predict(const std::vector<std::vector<T>>& x_test) {
    KNNResult<T> results;
    std::vector<int> predictions;

    std::vector<int> classes = this->stored_y_train;
    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());

    for (const auto& unknown_vector : x_test) {
        std::vector<double> distances;
        for (const auto& train_vector : this->stored_x_train) {
            distances.push_back(calculate_distance(unknown_vector, train_vector));
        }

        std::vector<int> indexes = argsort(distances);
        double power = 2.0 / (m - 1.0);

        std::unordered_map<int, double> test_membership;

        for (int c : classes) {
            double numerator = 0.0;
            double denominator = 0.0;

            for (int i = 0; i < k; i++) {
                int index = indexes[i];
                double d = std::max(distances[index], 1e-6);
                double weight = 1.0 / std::pow(d, power);
                numerator += stored_membership[c][index] * weight;
                denominator += weight;
            }

            test_membership[c] = numerator / denominator;
        }
        
        double total_membership = 0.0;
        for (auto&[c,u]: test_membership){
            total_membership += u;
        }
        if (total_membership > 0){
            for (auto&[c,u]: test_membership){
                test_membership[c] = u / total_membership;
            }
        }

        int prediction;
        double max_membership = 0;
        for (int c : classes) {
            if (test_membership[c] > max_membership) {
                max_membership = test_membership[c];
                prediction = c;
            }
        }

        results.predictions.push_back(prediction);
        results.memberships.push_back(test_membership);
    }

    return results;
}

#endif
