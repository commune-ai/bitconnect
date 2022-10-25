// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;


struct TokenState {
    address Address;
    string symbol;
    uint256 value;
    uint256 balance;
    uint256 ratio;
    int256 valueChange;
    int256 decimals;
    uint index;
}


struct PortfolioState {
    uint256 marketValue;
    uint256 depositValue;
    uint256 timestamp;
    uint256 blockNumber;
    string baseTokenSymbol;
}




