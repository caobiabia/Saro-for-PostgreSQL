select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND c.CreationDate>='2010-07-28 15:37:02'::timestamp  AND p.Score<=11  AND p.FavoriteCount>=0  AND p.FavoriteCount<=10  AND u.Reputation<=246  AND u.Views>=0  AND u.Views<=51  AND u.UpVotes>=0  AND u.UpVotes<=38  AND u.CreationDate>='2010-08-06 07:43:09'::timestamp;