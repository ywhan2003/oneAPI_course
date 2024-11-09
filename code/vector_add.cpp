#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
using namespace sycl;

constexpr int global_size = 32;
constexpr int local_size = 8;

template <typename T>
class vector_addition {
public:
    std::vector<T> add(std::vector<T> array1, std::vector<T> array2);

private:
    std::vector<T> add_array;
    void initialize_data(T (*data)[2], std::vector<T> array1, std::vector<T> array2, size_t size);
    std::vector<std::vector<T>> split_vector(const std::vector<T>& vec, size_t length);
};

template <typename T>
void vector_addition<T>::initialize_data(T (*data)[2], std::vector<T> array1, std::vector<T> array2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i][0] = array1[i];
        data[i][1] = array2[i];
    }
}

template <typename T>
std::vector<std::vector<T>> vector_addition<T>::split_vector(const std::vector<T>& vec, size_t length) {
    std::vector<std::vector<T>> result;
    size_t num_parts = (vec.size() + length - 1) / length;

    for (size_t i = 0; i < num_parts; ++i) {
        size_t start = i * length;
        size_t end = std::min(start + length, vec.size());
        result.push_back(std::vector<T>(vec.begin() + start, vec.begin() + end));
    }

    return result;
}

template <typename T>
std::vector<T> vector_addition<T>::add(std::vector<T> array1, std::vector<T> array2) {
    queue my_gpu_queue(sycl::cpu_selector_v);
    add_array.clear();

    auto split1 = split_vector(array1, global_size);
    auto split2 = split_vector(array2, global_size);
    int total_cnt = split1.size();

    int (*data)[2] = malloc_shared<int[2]>(global_size, my_gpu_queue);
    int *result = malloc_shared<int>(global_size, my_gpu_queue);

    for (size_t cnt = 0; cnt < total_cnt; cnt++) {
        size_t size = split1[cnt].size();
        
        initialize_data(data, split1[cnt], split2[cnt], size);

        nd_range<1> my_range{range{size}, range{local_size}};

        my_gpu_queue.submit([&](handler& h) {
            h.parallel_for(my_range, [=](nd_item<1> item) {
                int index = item.get_global_id(0);  // 获取全局ID
                result[index] = data[index][0] + data[index][1];
            });
        }).wait();

        printf("\nData Result\n");
        for(int i = 0; i < size; i++) {
            printf("%d, ", result[i]);
            add_array.push_back(result[i]);
        }
        printf("\nTask Done!\n");
    }
    free(data, my_gpu_queue);
    free(result, my_gpu_queue);
}

int main(int argc, char *argv[]) {
    std::vector<int> array1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> array2{3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    vector_addition<int> va;
    va.add(array1, array2);
    return 0;
}