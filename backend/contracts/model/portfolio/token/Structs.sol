// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;

import {PortfolioState} from "contracts/model/portfolio/Structs.sol";

struct NFTOwnerState {
    uint256[] tokenIds;
}

struct NFTContractState {
    uint256 ownerCount;
    uint256 tokenCount;
}

struct NFTState {
    uint256 depositValue;
    uint256 marketValue;
    uint256 lastUpdateBlock;
    PortfolioState portfolio;
}


