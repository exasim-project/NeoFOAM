// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <source_location>
#include <iostream>
#include <string>

namespace NeoFOAM {

	class error {
		public:	

		/**
		 * @brief Exit the program with an error message.
		 * 
		 * @param errNo The error number to exit with.
		 * @param location Default argument for the location of the error.
		 */
		void exit(const int errNo = 1, const std::source_location location = std::source_location::current()) {
			std::cout << "Error: " << errNo << '\n'
		        		<< "At: " << location.file_name() << '\n'
						    << "Line: " << location.line() << '\n'
						    << "Function: " << location.function_name() << '\n';
				std::exit(errNo);
			};

			error(std::string){};
	};

	extern error FatalError;
}
