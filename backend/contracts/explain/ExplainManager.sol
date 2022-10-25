// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.7;
import {ExplainSchema} from "contracts/explain/Structs.sol";
import "contracts/utils/Strings.sol";

contract ExplainManager {
    using Strings for string;

    ExplainSchema[] public explainModules ;
    
    mapping(string=>uint) public name2id ;

    // add explainer object to Modules List
    // replaces existing name
    function addExplain(string memory name , string memory explainURI) public  {
        
        if (name2id[name] > 0) {
            removeExplain(name);
        }
        explainModules.push(ExplainSchema(name, explainURI));
        // id is +1 greater than index
        name2id[name] = explainModules.length;
    }


    // remove explain object
    function removeExplain(string memory name)  public {
        require(name2id[name]>0, "The name does not exist");

        // update the list of names
        uint _id = name2id[name];
        ExplainSchema storage lastExplainModule = explainModules[explainModules.length-1];
        name2id[lastExplainModule.name] = _id;
        explainModules[_id-1] = lastExplainModule;
        explainModules.pop();
        delete name2id[name];

    }

    function getExplainer(string memory name) external view returns(ExplainSchema memory ){
        require(name2id[name]>0, "The name does not exist");
        return explainModules[name2id[name]];
    } 
    function getExplainers() external view returns(ExplainSchema[] memory ){
        require(explainModules.length>0, "Empty Explainer Modules");
        return explainModules;
    } 
}
