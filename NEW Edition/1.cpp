#include <Python.h>
#include <iostream>
void fengzhuang()
{
        Py_Initialize();

    // 确保 Python 在当前目录查找模块
    PyRun_SimpleString("import sys; sys.path.insert(0, '')");

    PyObject* pName = PyUnicode_DecodeFSDefault("detect_dashboard_trt");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "run_dashboard");  // 函数名
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject* pValue = PyObject_CallObject(pFunc, nullptr);
            if (pValue) {
                // 假设 Python 函数返回字符串
                std::cout << "Python 返回: " << PyUnicode_AsUTF8(pValue) << std::endl;
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
            }
            Py_DECREF(pFunc);
        } else {
            std::cerr << "找不到函数 run_dashboard 或不可调用\n";
            PyErr_Print();
        }
        Py_DECREF(pModule);
    } else {
        std::cerr << "无法加载模块 dashboard_infer\n";
        PyErr_Print();
    }

    Py_Finalize();
}
int main() {
    fengzhuang();
    return 0;
}
