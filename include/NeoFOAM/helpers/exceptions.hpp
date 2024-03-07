// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

namespace NeoFOAM
{
/**
 * The Error class is used to report exceptional behaviour in library
 * functions. NeoFOAM uses C++ exception mechanism to this end, and the
 * Error class represents a base class for all types of errors. The exact list
 * of errors which could occur during the execution of a certain library
 * routine is provided in the documentation of that routine, along with a short
 * description of the situation when that error can occur.
 * During runtime, these errors can be detected by using standard C++ try-catch
 * blocks, and a human-readable error description can be obtained by calling
 * the Error::what() method.
 *
 * As an example, trying to compute a matrix-vector product with arguments of
 * incompatible size will result in a DimensionMismatch error, which is
 * demonstrated in the following program.
 *
 * ```cpp
 * #include <ginkgo.h>
 * #include <iostream>
 *
 * using namespace gko;
 *
 * int main()
 * {
 *     auto omp = create<OmpExecutor>();
 *     auto A = randn_fill<matrix::Csr<float>>(5, 5, 0f, 1f, omp);
 *     auto x = fill<matrix::Dense<float>>(6, 1, 1f, omp);
 *     try {
 *         auto y = apply(A, x);
 *     } catch(Error e) {
 *         // an error occurred, write the message to screen and exit
 *         std::cout << e.what() << std::endl;
 *         return -1;
 *     }
 *     return 0;
 * }
 * ```
 *
 * @ingroup error
 */
class Error : public std::exception
{
public:

    /**
     * Initializes an error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param what  The error message
     */
    Error(const std::string& file, int line, const std::string& what)
        : what_(file + ":" + std::to_string(line) + ": " + what)
    {}

    /**
     * Returns a human-readable string with a more detailed description of the
     * error.
     */
    virtual const char* what() const noexcept override { return what_.c_str(); }

private:

    const std::string what_;
};


/**
 * DimensionMismatch is thrown if an operation is being applied to containers of
 * incompatible size.
 */
class DimensionMismatch : public Error
{
public:

    /**
     * Initializes a dimension mismatch error.
     *
     * @param file  The name of the offending source file
     * @param line  The source code line number where the error occurred
     * @param func  The function name where the error occurred
     * @param length_a  The size of the first container
     * @param length_b  The size of the second container
     * @param clarification  An additional message describing the error further
     */
    DimensionMismatch(
        const std::string& file, int line, const std::string& func, size_t length_a, size_t length_b, const std::string& clarification
    )
        : Error(file, line, func + ": Trying to perform binary operation " + " " + std::to_string(length_a) + ", " + std::to_string(length_b) + " " + clarification)
    {}
};


/**
 * Asserts that `_op1` and `_op2` have the same length.
 *
 * @throw DimensionMismatch  if `_op1` and `_op2` differ in the number of
 *                           rows or columns
 */
#define NeoFOAM_ASSERT_EQUAL_LENGTH(_op1, _op2)                                                 \
    if (_op1.size() != _op2.size())                                                             \
    {                                                                                           \
        throw ::NeoFOAM::DimensionMismatch(                                                     \
            __FILE__, __LINE__, __func__, _op1.size(), _op2.size(), "expected equal dimensions" \
        );                                                                                      \
    }
}
