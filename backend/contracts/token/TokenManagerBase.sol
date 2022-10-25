// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.7;

/*
@title myVault
@license GNU GPLv3
@author James Bachini
@notice A vault to automate and decentralize a long term donation strategy
*/
// import "@openzeppelin/contracts/token/ERC20/extensions/IERC20Metadata.sol";
import "contracts/token/ERC20/ERC20.sol";
// interface IERC20Extented is IERC20 {
//     function decimals() view returns (uint8);
// }


contract TokenManagerBase {

    mapping(bytes32=>address) public symbol_address_map;
    mapping(bytes32=>ERC20) public symbol_token_map;


    bytes32[] public token_symbol_list ;
    address[] public token_address_list ;

    uint public token_count;


    
}