extern crate crdts;

use crdts::{CmRDT, CvRDT, Map, Orswot};

fn main() {
    let mut friend_map: Map<&str, Orswot<&str, u8>, u8> = Map::new();

    let read_ctx = friend_map.read_ctx();
    friend_map.apply(
        friend_map.update("bob", read_ctx.derive_add_ctx(1), |set, ctx| {
            set.add("janet", ctx)
        }),
    );

    let mut friend_map_on_2nd_device = friend_map.clone();

    // the map on the 2nd devices adds 'erik' to `bob`'s friends
    friend_map_on_2nd_device.apply(friend_map_on_2nd_device.update(
        "bob",
        friend_map_on_2nd_device.read_ctx().derive_add_ctx(2),
        |set, c| set.add("erik", c),
    ));

    // Meanwhile, on the first device we remove
    // the entire 'bob' entry from the friend map.
    friend_map.apply(friend_map.rm("bob", friend_map.read_ctx().derive_rm_ctx()));

    assert!(friend_map.get(&"bob").is_none());

    // once these two devices synchronize...
    let friend_map_snapshot = friend_map.clone();
    let friend_map_on_2nd_device_snapshot = friend_map_on_2nd_device.clone();

    friend_map.merge(friend_map_on_2nd_device_snapshot);
    friend_map_on_2nd_device.merge(friend_map_snapshot);
    assert_eq!(friend_map, friend_map_on_2nd_device);

    // ... we see that "bob" is present but only
    // contains `erik`.
    //
    // This is because the `erik` entry was not
    // seen by the first device when it deleted
    // the entry.
    let bobs_friends = friend_map
        .get(&"bob")
        .map(|set| set.read().val)
        .map(|hashmap| hashmap.into_iter().collect::<Vec<_>>());

    assert_eq!(bobs_friends, Some(vec!["erik"]));
}
