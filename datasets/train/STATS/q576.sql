select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND p.ViewCount>=0  AND p.ViewCount<=9951  AND p.CommentCount>=0  AND p.CommentCount<=18  AND p.FavoriteCount>=0  AND u.Reputation<=607  AND u.UpVotes>=0  AND u.CreationDate>='2010-07-19 19:09:39'::timestamp  AND u.CreationDate<='2014-09-10 18:15:53'::timestamp  AND v.VoteTypeId=2  AND v.BountyAmount<=50;