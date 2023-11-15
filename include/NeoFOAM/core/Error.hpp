// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <string>

namespace NeoFOAM {

	class error {
		public:
		        void exit(const int errNo = 1) {};

			error(std::string){};
	};

	extern error FatalError;
}
