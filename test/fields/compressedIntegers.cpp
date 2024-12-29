// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors

#define CATCH_CONFIG_RUNNER // Define this before including catch.hpp to create
                            // a custom main
#include <bit>
#include <bitset>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>

#include "NeoFOAM/fields/segmentedField.hpp"
#include "NeoFOAM/core/primitives/label.hpp"
#include <Kokkos_Core.hpp>

std::vector<std::uint8_t> uint32ToVector(std::uint32_t value)
{
    value <<= 3; // Shift the value 3 bits to the left
    std::vector<std::uint8_t> result(4);
    result[0] = static_cast<std::uint8_t>(value & 0xFF);
    result[1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    result[2] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
    result[3] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
    return result;
}

std::uint32_t decodeVectorToUint32(const std::vector<std::uint8_t>& vec)
{
    if (vec.size() < 4)
    {
        throw std::invalid_argument("Vector size must be at least 4 bytes");
    }
    return (static_cast<std::uint32_t>(vec[0]) << 0) | (static_cast<std::uint32_t>(vec[1]) << 8)
         | (static_cast<std::uint32_t>(vec[2]) << 16) | (static_cast<std::uint32_t>(vec[3]) << 24);
}

TEST_CASE("compressedIntegers")
{
    // NeoFOAM::Executor exec = GENERATE(NeoFOAM::Executor(NeoFOAM::SerialExecutor {}),
    //                                   NeoFOAM::Executor(NeoFOAM::CPUExecutor {}),
    //                                   NeoFOAM::Executor(NeoFOAM::GPUExecutor {})
    // );
    NeoFOAM::Executor exec = NeoFOAM::Executor(NeoFOAM::SerialExecutor {});

    std::string execName = std::visit([](auto e) { return e.print(); }, exec);

    SECTION("compress " + execName)
    {
        // for (unsigned x {}; x != 010; ++x)
        // {
        //     std::cout << "bit_width( " << std::bitset<4> {x} << " ) = " << std::bit_width(x)
        //               << '\n';
        // }
        // std::uint8_t x1 = 8;
        // std::cout << "bit_width( " << std::bitset<8> {x1} << " ) = " << std::bit_width(x1)
        //     << '\n';
        // std::int8_t x2 = -1;
        // std::uint8_t absx2 = std::abs(x2);
        // std::cout << "bit_width( " << std::bitset<8> {x2} << " ) = " << std::bit_width(absx2) <<
        // '\n'; std::int8_t x3 = 127; std::uint8_t absx3 = std::abs(x3); std::cout << "bit_width( "
        // << std::bitset<8> {x3} << " ) = " << std::bit_width(absx3) << '\n';
        // NeoFOAM::Field<std::uint8_t> values(exec, 1,8);
        // auto hostValues = values.copyToHost();
        // REQUIRE(values.size() == 1);
        // auto valuesSpan = values.span();
        // NeoFOAM::parallelFor(
        //     exec, {0, 1}, KOKKOS_LAMBDA(const size_t i) { valuesSpan[i] =
        //     std::bit_width(valuesSpan[i]); }
        // );
        // hostValues = values.copyToHost();
        // REQUIRE(hostValues[0] == 4);
        // REQUIRE(false);
        // std::vector<std::uint8_t> valuesVec = {1, 0, 0, 0, 0, 0, 0, 0};


        // std::uint32_t compressedValue = *reinterpret_cast<std::uint32_t*>(valuesVec.data());
        // REQUIRE(compressedValue == 1);

        std::vector<std::uint8_t> valuesVec = {1, 0, 0, 0, 0, 0, 0, 0};

        std::uint32_t compressedValue = *reinterpret_cast<std::uint32_t*>(valuesVec.data());
        REQUIRE(compressedValue == 1);
        std::cout << "Bit representation of compressedValue: " << std::bitset<32>(compressedValue)
                  << std::endl;

        uint32_t num = 1;
        uint8_t* numPtr = reinterpret_cast<uint8_t*>(&num);

        if (numPtr[0] == 1)
        {
            std::cout << "Little-endian" << std::endl;
        }
        else
        {
            std::cout << "Big-endian" << std::endl;
        }

        std::uint32_t value = 1;
        std::vector<std::uint8_t> valuesVec2 = uint32ToVector(value);

        // Print the vector to verify the conversion
        std::cout << "Vector representation of value: " << std::endl;
        for (std::uint8_t byte : valuesVec2)
        {
            std::cout << "byte: " << static_cast<int>(byte) << std::endl;
        }
        // std::cout << std::endl;

        // Decode the vector back to uint32_t
        std::uint32_t decodedValue = decodeVectorToUint32(valuesVec2);
        std::cout << "decodedValue: " << decodedValue << std::endl;

        // REQUIRE(decodedValue == value);
        std::cout << "Bit representation of decodedValue: " << std::bitset<32>(decodedValue)
                  << std::endl;

        REQUIRE(false);
    }
}
