// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  queue my_gpu_queue( gpu_selector{} );

  std::cout << "Selected GPU device: " <<
    my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

  return 0;
}

