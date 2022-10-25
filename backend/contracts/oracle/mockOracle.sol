// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC20/ERC20.sol)

pragma solidity ^0.8.0;

import "interfaces/token/ERC20/IERC20.sol";
import "contracts/utils/context/Context.sol";
import "contracts/utils/access/AccessControlAdapter.sol";


contract MockOracle  {
 
    constructor(string memory name_, string memory symbol_) {
        
    }
}