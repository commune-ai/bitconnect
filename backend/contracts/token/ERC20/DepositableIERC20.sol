// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC20/IERC20.sol)

pragma solidity ^0.8.3;

import '@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol';
// import '@openzeppelin/contracts/token/ERC20/IERC20.sol'

/**
 * @dev Interface of the ERC20 standard as defined in the EIP.
 */
interface DepositableIERC20 is IERC20Metadata {

    function deposit() external payable;
}
