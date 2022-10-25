// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts v4.4.1 (token/ERC721/ERC721.sol)

pragma solidity ^0.8.0;
import "interfaces/model/portfolio/token/IDepositNFT.sol";

/**
 * @dev Implementation of https://eips.ethereum.org/EIPS/eip-721[ERC721] Non-Fungible Token Standard, including
 * the Metadata extension, but not including the Enumerable extension, which is available separately as
 * {ERC721Enumerable}.
 */
contract DepositNFTAdapter {

    IDepositNFT public depositNFT;
    function connectNFT(address tokenAddress) public {
        depositNFT = IDepositNFT(tokenAddress);
    }
}
