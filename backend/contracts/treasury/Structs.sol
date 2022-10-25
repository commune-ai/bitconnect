// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

struct UserState {
    address user;
    address[] assets;
    uint256 lastUpdateBlock;
}


struct AssetState {
    address asset; // address of the asset
    string name; // name of the asset
    string mode; //what is the mode of the asset [ERC20,ERC721,Treasury]
    uint256 balance; // balance 
    uint256 value; // what is the value 
    address valueOracle; // what is the current price
    bool liquid; // is the asset liquid
    uint256 lastUpdateBlock; // when was the value updated
    bytes metaData; // additional metadata
}

