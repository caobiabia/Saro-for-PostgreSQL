select  count(*) from postHistory as ph,          posts as p,  		users as u,  		badges as b  where u.Id = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = b.UserId  AND p.PostTypeId=2  AND p.ViewCount<=1700  AND p.CommentCount>=0  AND p.FavoriteCount<=12  AND u.Reputation<=966;