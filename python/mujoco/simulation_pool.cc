#include <iostream>
#include <optional>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <mujoco/mujoco.h>
#include "errors.h"
#include "raw.h"
#include "structs.h"
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace mujoco::python {

namespace {

namespace py = ::pybind11;

class SimulationPool {
    public:
        SimulationPool(const MjModelWrapper& py_mj_model, int nroll, int nworkers);
        ~SimulationPool();
        void step();
        py::memoryview getState();
        py::memoryview getSubtree_com();
        void setControl(py::array_t<mjtNum> new_control);
        void setReset(py::array_t<bool> new_reset_mask);
    private:
        void threadLoop();
        void stopWorkers();

        const mjModel* model;
        std::vector<mjData*> data;
        std::vector<std::thread> workers;
        int nstate;
        int ncontrol;
        mjtNum* control;
        mjtNum* state;
        mjtNum* subtree_com;
        std::vector<bool> resetMask;

        std::mutex nextModelMutex;
        std::condition_variable nextModelAvailable;
        unsigned long long nextModelId;

        std::mutex completedMutex;
        std::condition_variable allCompleted;
        unsigned long long completedModels;

        bool shutdownWorkers = false;
};

SimulationPool::SimulationPool(const MjModelWrapper& py_mj_model, int nroll, int nworkers) : model(py_mj_model.get()) {
    for(int i=0; i<nroll; i++) {
        data.push_back(mj_makeData(model));
    }
    nstate = mj_stateSize(model, mjSTATE_FULLPHYSICS);
    ncontrol = mj_stateSize(model, mjSTATE_CTRL);
    state = new mjtNum[nroll * nstate];
    mju_zero(state, nroll * nstate);
    control = new mjtNum[nroll * ncontrol];
    mju_zero(control, nroll * ncontrol);
    subtree_com = new mjtNum[nroll * model->nbody * 3];
    mju_zero(subtree_com, nroll * model->nbody * 3);
    resetMask = std::vector<bool>(nroll, false);
    for(int i=0; i<nworkers; i++) {
        workers.emplace_back(std::thread(&SimulationPool::threadLoop, this));
    }
}

SimulationPool::~SimulationPool() {
    stopWorkers();
    for(mjData* d : data) {
        mj_deleteData(d);
    }
    delete control;
    delete state;
    delete subtree_com;
}

void SimulationPool::threadLoop() {
    while(true) {
        unsigned long long modelId;
        {
            std::unique_lock<std::mutex> lock(nextModelMutex);
            nextModelAvailable.wait(lock, [this] {
                return nextModelId<data.size() || shutdownWorkers;
            });
            if(shutdownWorkers) return;
            if(nextModelId >= data.size()) continue;
            modelId = nextModelId;
            nextModelId++;
        }
        mjData* data_obj = data[modelId];
        if(resetMask[modelId]) {
            mj_resetData(model, data_obj);
            mju_copy(data_obj->ctrl, control + modelId*ncontrol, ncontrol);
        } else {
            mju_copy(data_obj->ctrl, control + modelId*ncontrol, ncontrol);
            mj_step(model, data_obj);
        }
        mj_getState(model, data_obj, state + modelId*nstate, mjSTATE_FULLPHYSICS);
        mju_copy(subtree_com + modelId*model->nbody * 3, data_obj->subtree_com, model->nbody * 3);
        {
            std::unique_lock<std::mutex> lock(completedMutex);
            completedModels++;
            if(completedModels >= data.size()) {
                allCompleted.notify_all();
            }
        }
    }
}

void SimulationPool::step() {
    {
        std::unique_lock<std::mutex> lock(nextModelMutex);
        std::unique_lock<std::mutex> second_lock(completedMutex);
        nextModelId = 0;
        completedModels = 0;
    }
    nextModelAvailable.notify_all();
    while (true) {
        std::unique_lock<std::mutex> lock(completedMutex);
        allCompleted.wait(lock, [this] {
            return completedModels>=data.size();
        });
        if (completedModels>=data.size()) return;
    }
}

void SimulationPool::stopWorkers() {
    {
        std::unique_lock<std::mutex> lock(nextModelMutex);
        shutdownWorkers = true;
    }
    nextModelAvailable.notify_all();
    for (std::thread& t : workers) {
        t.join();
    }
    workers.clear();
}

py::memoryview SimulationPool::getState() {
    const long int nroll = data.size();
    const long int nstate = this->nstate;
    return py::memoryview::from_buffer(state, {nroll, nstate},
                                       {sizeof(mjtNum)*nstate, sizeof(mjtNum)}, true);
}

void SimulationPool::setControl(py::array_t<mjtNum> new_control) {
    if(new_control.ndim() != 2) {
        throw py::value_error("Argument `control` must be 2-dim array.");
    }
    if(new_control.shape()[0] != data.size() || new_control.shape()[1] != ncontrol) {
        throw py::value_error("Argument `control` must have dimensions nroll x ncontrol .");
    }
    //Don't know stride etc of new_control, so this is probably safest way to copy?
    auto r = new_control.unchecked<2>();
    for(int i=0; i<data.size(); i++) {
        for(int j=0; j<ncontrol; j++) {
            control[i*ncontrol + j] = r(i, j);
        }
    }
}

void SimulationPool::setReset(py::array_t<bool> new_reset_mask) {
    if(new_reset_mask.ndim() != 1 || new_reset_mask.shape()[0] != data.size()) {
        throw py::value_error("Argument `reset_mask` must be an array with nroll elements.");
    }
    //Don't know stride etc of new_reset_mask, so this is probably safest way to copy?
    auto r = new_reset_mask.unchecked<1>();
    for(int i=0; i<data.size(); i++) {
        resetMask[i] = r(i);
    }
}

py::memoryview SimulationPool::getSubtree_com() {
    const long int nroll = data.size();
    const long int three = 3;
    const long int nbody = model->nbody;
    return py::memoryview::from_buffer(subtree_com, {nroll, nbody, three},
                                       {sizeof(mjtNum)*nbody*three, sizeof(mjtNum)*three, sizeof(mjtNum)},
                                       true);
}

PYBIND11_MODULE(_simulation_pool, pymodule) {
    namespace py = ::pybind11;
    py::class_<SimulationPool>(pymodule, "SimulationPool")
        .def(py::init<const MjModelWrapper&, int, int>(), py::arg("model"), py::arg("nroll"), py::arg("nworkers"))
        .def("step", &SimulationPool::step, "Step all environments one step using the threadpool.")
        .def("getState", &SimulationPool::getState, "Get a memory view of the state (nroll x nstate).")
        .def("setControl", &SimulationPool::setControl, "Set the control inputs.", py::arg("new_control"))
        .def("setReset", &SimulationPool::setReset, "Flag some simulations for resetting. Next time `step()` is called, these simulations will reset instead of stepping.", py::arg("reset_mask"))
        .def("getSubtree_com", &SimulationPool::getSubtree_com, "Get a memory view of the subtree bodies center-of-mass (nroll x nbody x 3).");
}

}

}
