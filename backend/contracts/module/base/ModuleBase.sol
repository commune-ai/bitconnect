// // SPDX-License-Identifier: MIT OR Apache-2.0
// pragma solidity ^0.8.3;

// import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
// import "interfaces/utils/access/IAccessControl.sol";
// import {CreatorState, ItemState} from "./Structs.sol";

// contract MarketBase is ReentrancyGuard {
  
//   address payable owner;
//   uint256 nextItemId = 0;
//   string public name;
//   mapping(address=>CreatorState) creatorState;
//   mapping(uint256 => ItemState) itemState;
//   mapping(uint256=>uint256) public id2index;
//   mapping(address=>uint256) public item2id;
//   uint256[] public itemIds;

//   constructor(string memory _name) {
//     owner = payable(msg.sender);
//     name = _name;
//   }


//   event ItemCreated (
//     uint indexed itemId,
//     address indexed nftContract,
//     string  name, 
//     string  category
//     );

//   /* Places an item for sale on the marketplace */
//   function addItem(address modelContract) public payable nonReentrant {
//     if (item2id[modelContract]==0) {
//     nextItemId ++;
//     uint256 itemId = nextItemId;

//     // ensure msg.sender has owner role for ModelNFT contract
//     // require(ModelNFT(nftContract).hasRole("owner", msg.sender),
//     //          "msg.sender does not have owner priveledges");

//     itemIds.push(nextItemId);
//     id2index[itemId] = itemIds.length-1;
//     creatorState[msg.sender].itemIds.push(itemId);
//     itemState[itemId] =  ItemState(
//       itemId,
//       modelContract,
//       msg.sender
//       );
//     item2id[modelContract] = itemId;  
//     }
//   }

//   function removeItem(uint256 itemId) public payable nonReentrant {

//     // ensure msg.sender has owner role for ModelNFT contract
//     // require(ModelNFT(nftContract).hasRole("owner", msg.sender),
//     //          "msg.sender does not have owner priveledges");

//     if (item2id[itemState[itemId].item]!=0) {
//       uint256[] storage creatorTokenIds = creatorState[msg.sender].itemIds;
//       for (uint i; i<creatorTokenIds.length;i++) {
//         if (creatorTokenIds[i] == itemId) {
//           creatorTokenIds[i] = creatorTokenIds[creatorTokenIds.length-1];
//           creatorTokenIds.pop();
//         }
//       }

//       // swap last id with current id
//       id2index[itemIds[itemIds.length-1]] = id2index[itemId];
//       itemIds[id2index[itemId]] = itemIds[itemIds.length-1];
//       delete id2index[itemId];
//       itemIds.pop();

//       delete item2id[itemState[itemId].item];
//       delete itemState[itemId];
      
    
//     }

//   }
  
//   /* Returns all unsold market items */
//   function listitemState() public view returns (ItemState[] memory) {
//     uint itemCount = nextItemId;

//     ItemState[] memory items = new ItemState[](itemCount);
//     for (uint i = 0; i < itemIds.length; i++) {
//         ItemState storage currentItem = itemState[itemIds[i]];
//         items[i] = currentItem;      
//     }
//     return items;
//   }
//   function listMyItems() public view returns (ItemState[] memory) {
//     uint256[] storage creatorItemIds = creatorState[msg.sender].itemIds;
//     ItemState[] memory items = new ItemState[](creatorItemIds.length);
//     for (uint i = 0; i < creatorItemIds.length; i++) {
//         items[i] = itemState[itemIds[i]];      
//     }
//     return items;
//   }
//   /* Returns onlyl items that a user has purchased */

//   /* Returns only items a user has created */
// }