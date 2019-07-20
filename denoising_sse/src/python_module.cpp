#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>
#include <iostream>

cv::Mat denoising(unsigned char *input_img, int width, int height, int channel, int iteration, bool use_avx);
void denoising_test(const char* filepath);

namespace pbcvt {

    using namespace boost::python;
    using namespace std;

/**
 * @brief Example function. Basic inner matrix product using explicit matrix conversion.
 * @param left left-hand matrix operand (NdArray required)
 * @param right right-hand matrix operand (NdArray required)
 * @return an NdArray representing the dot-product of the left and right operands
 */
    PyObject *dot(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        // Check that the 2-D matrices can be legally multiplied.
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(result);
        return ret;
    }
/**
 * @brief Example function. Simply makes a new CV_16UC3 matrix and returns it as a numpy array.
 * @return The resulting numpy array.
 */

	PyObject* makeCV_16UC3Matrix(){
		cv::Mat image = cv::Mat::zeros(240, 320, CV_16UC3);
		PyObject* py_image = pbcvt::fromMatToNDArray(image);
		return py_image;
	}

//
/**
 * @brief Example function. Basic inner matrix product using implicit matrix conversion.
 * @details This example uses Mat directly, but we won't need to worry about the conversion in the body of the function.
 * @param leftMat left-hand matrix operand
 * @param rightMat right-hand matrix operand
 * @return an NdArray representing the dot-product of the left and right operands
 */
    cv::Mat dot2(cv::Mat leftMat, cv::Mat rightMat) {
        auto c1 = leftMat.cols, r2 = rightMat.rows;
        if (c1 != r2) {
            PyErr_SetString(PyExc_TypeError,
                            "Incompatible sizes for matrix multiplication.");
            throw_error_already_set();
        }
        cv::Mat result = leftMat * rightMat;

        return result;
    }

    cv::Mat curvature_filter_sse(cv::Mat input_mat, int iteration)
    {
        IplImage input_img = IplImage(input_mat);
        int channel = 3;
        cv::Mat out_img = denoising((unsigned char*) input_img.imageDataOrigin, input_img.width, input_img.height, channel, iteration, false);
        return out_img;
    }

    cv::Mat curvature_filter_avx(cv::Mat input_mat, int iteration)
    {
        IplImage input_img = IplImage(input_mat);
        int channel = 3;
        cv::Mat out_img = denoising((unsigned char*) input_img.imageDataOrigin, input_img.width, input_img.height, channel, iteration, true);
        return out_img;
    }

    // cv::Mat curvature_filter_ori(cv::Mat input_mat, int iteration)
    // {
    //     IplImage input_img = IplImage(input_mat);
    //     int channel = 3;
    //     cv::Mat out_img = denoising((unsigned char*) input_img.imageDataOrigin, input_img.width, input_img.height, channel, iteration);
    //     return out_img;
    // }

    void test(const char* filepath)
    {
        denoising_test(filepath);
    }

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        matFromNDArrayBoostConverter();

        //expose module-level functions
        def("dot", dot);
        def("dot2", dot2);
		def("makeCV_16UC3Matrix", makeCV_16UC3Matrix);

		//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
        //"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
        def("curvature_filter_sse", curvature_filter_sse);
        def("curvature_filter_avx", curvature_filter_avx);
        def("test", test);

    }

} //end namespace pbcvt
